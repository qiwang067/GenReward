import argparse
import functools
import os
import pathlib
import sys
# wandb an logdir
# import wandb
import datetime
import socket
import swanlab as wandb
# import wandb

import metaworld_envs as meta_env

# swanlab.sync_wandb()

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import eval_tools as tools
# import envs.wrappers_eval as wrappers_eval

import wrappers_eval
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    # breakpoint()
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers_eval.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers_eval.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers_eval.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers_eval.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers_eval.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers_eval.OneHotAction(env)
    elif suite == "metaworld":  # Define meta-world in make-env environment
        task = "-".join(task.split("_"))
        env = wrappers_eval.MetaWorld(
            task,
            config.seed,
            config.action_repeat,
            config.size,
            config.camera,
        )
        env = wrappers_eval.NormalizeActions(env)
    elif suite == "metaworldgc":  # Define meta-world in make-env environment
        task = "-".join(task.split("_"))
        # rank = 0
        # env_id = "button-press-v2-goal-hidden"
        # env = meta_env.MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)
        # breakpoint()
        if mode == "eval":
            env = wrappers_eval.MetaWorld(
            task,
            config.seed,
            config.action_repeat,
            config.size,
            config.camera,
            )
        else:
            env = wrappers_eval.MetaWorldGC(
                task,
                config.seed,
                config.action_repeat,
                config.size,
                config.camera,
                video_path=config.video_path,
                human=True,
                ts=True,
                reward_type=config.reward_type
            )
        env = wrappers_eval.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers_eval.TimeLimit(env, config.time_limit)
    env = wrappers_eval.SelectAction(env, key="action")
    env = wrappers_eval.UUID(env)
    if suite == "minecraft":
        env = wrappers_eval.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    
    # breakpoint()
    # Set log directory
    # if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
    #     # If checkpoint path is specified, infer logdir from it
    #     checkpoint_path = pathlib.Path(config.checkpoint_path).expanduser()
    #     logdir = checkpoint_path.parent
        
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
        
    
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    wandb_device = socket.gethostname()+':'+config.device[-1]

    '''
    wandb initialization (for evaluation mode only)
    '''
    config.experiment_name = str(logdir) + "_eval"
    run_dir = pathlib.Path("results") / config.project_name / config.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    config.description = wandb_device + " " + config.description + " [EVAL ONLY]"
    wandb.init(
        project=config.project_name,
        notes=wandb_device,
        mode=config.wandb_mode,
        description=config.description,
        name=config.experiment_name,
        logdir=str(run_dir),
        reinit=True)
    
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    
    # For evaluation only, no training data needed
    logger = tools.Logger(logdir, 0)

    print("Create eval envs.")
    # Load evaluation episodes only
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    
    make = lambda mode, id: make_env(config, mode, id)
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        eval_envs = [Damy(env) for env in eval_envs]
    
    acts = eval_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    print("Initialize agent.")
    # Create an empty dataset for initialization
    eval_dataset = make_dataset(eval_eps, config)
    
    agent = Dreamer(
        eval_envs[0].obs_space,
        eval_envs[0].action_space,
        config,
        logger,
        eval_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    
    # Load checkpoint
    
    checkpoint_file = None
    if hasattr(config, 'checkpoint_path') and config.checkpoint_path:
        checkpoint_file = pathlib.Path(config.checkpoint_path)
    else:
        checkpoint_file = logdir / "latest.pt"
    # breakpoint()
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        if "optims_state_dict" in checkpoint:
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")

    # Perform evaluation only
    print("Start evaluation.")
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(
        eval_policy,
        eval_envs,
        eval_eps,
        config.evaldir,
        logger,
        is_eval=True,
        episodes=config.eval_episode_num,
        save_dir=logdir,
        save_frame_every=getattr(config, 'save_frame_every', 0),
    )
    
    if config.video_pred_log:
        video_pred = agent._wm.video_pred(next(eval_dataset))
        logger.video("eval_openl", to_np(video_pred))
    
    logger.write()
    print("Evaluation completed.")
    
    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    # Add checkpoint path parameter
    # parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    # # Add checkpoint_path parameter again to ensure it's parsed correctly
    # parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")
    
    main(parser.parse_args(remaining))
    wandb.finish()