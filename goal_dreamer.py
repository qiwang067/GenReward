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
import tools
# import envs.wrappers as wrappers
import re

import wrappers
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

def make_reward_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, 2)
    dataset = tools.from_generator(generator, config.batch_size * config.batch_length)
    return dataset

def _make_calvin_reward_specs(config):
    """
     config  rewards.spec  wrappers.CalvinGC 
     config 
      reward_case              : fb / vae / s3d / iv2 / vaefb
      video_path               : 
      keep, interval, reward_weight
      fb_interval, fb_keep, fb_weight
      fb_config                : dict,  FBReward
    """
    specs = []
    rc = getattr(config, "reward_case", None)
    if not rc:
        return specs

    target_video = getattr(config, "video_path", None)
    base_keep = int(getattr(config, "keep", 128))
    base_interval = int(getattr(config, "interval", 128))
    base_weight = float(getattr(config, "reward_weight", 1.0))
    crop = getattr(config, "crop", True)
    device = getattr(config, "device", None)

    if rc == "vaefb":
        specs.append(
            {
                "type": "vae",
                "target_video_path": target_video,
                "keep": base_keep,
                "interval": base_interval,
                "weight": base_weight,
                "crop": crop,
                "size": getattr(config, "vae_size", (480, 480)),
                "device": device,
            }
        )
        specs.append(
            {
                "type": "fb",
                "target_video_path": target_video,
                "keep": int(getattr(config, "fb_keep", base_keep)),
                "interval": int(getattr(config, "fb_interval", base_interval)),
                "weight": float(getattr(config, "fb_weight", base_weight)),
                "device": device,
                "fb_config": getattr(config, "fb_config", None),
            }
        )
    elif rc == "fb":
        specs.append(
            {
                "type": "fb",
                "target_video_path": target_video,
                "keep": int(getattr(config, "fb_keep", base_keep)),
                "interval": int(getattr(config, "fb_interval", base_interval)),
                "weight": base_weight,
                "device": device,
                "fb_config": getattr(config, "fb_config", None),
            }
        )
    else:
        specs.append(
            {
                "type": rc,
                "target_video_path": target_video,
                "keep": base_keep,
                "interval": base_interval,
                "weight": base_weight,
                "crop": crop,
                "size": getattr(config, "vae_size", (480, 480)),
                "device": device,
            }
        )
    return specs

def make_env(config, mode, id):
    # breakpoint()
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
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
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "metaworld":
        task = "-".join(task.split("_"))
        env = wrappers.MetaWorld(
            task,
            config.seed,
            config.action_repeat,
            config.size,
            config.camera,
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "metaworldgc":
        task = "-".join(task.split("_"))
        # rank = 0
        # env_id = "button-press-v2-goal-hidden"
        # env = meta_env.MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)
        # breakpoint()
        if mode == "train":
            # breakpoint()
            # config.fb_config['device']=config.device

            # Select appropriate config based on reward_case
            if config.reward_case == 'diffusion':
                reward_config = config.diffusion_config if hasattr(config, 'diffusion_config') else config.fb_config
            else:
                reward_config = config.fb_config

            env = wrappers.MetaWorldGC(
                task,
                config.sparse_interval,
                config.alpha,
                config.beta_scale,
                config.device,
                config.crop,
                config.seed,
                config.action_repeat,
                config.size,
                config.camera,
                text_string=config.text_string,
                video_path=config.video_path,
                human=True,
                ts=True,
                reward_type=config.reward_type,
                reward_case=config.reward_case,
                fb_config=reward_config,
                use_vae_fb=False,
                viper_config=config.viper_config,
                tadpole_config=config.tadpole_config if hasattr(config, 'tadpole_config') else None,
            )
            # breakpoint()
        else:
            env = wrappers.MetaWorld(
            task,
            config.seed,
            config.action_repeat,
            config.size,
            config.camera,
            )
        env = wrappers.NormalizeActions(env)
    elif suite == "robodeskgc":
        env = wrappers.RoboDeskGC(
            task,
            reward_case=config.reward_case,
            reward_type=config.reward_type,
            alpha=config.alpha,
            device=config.device,
            video_path=config.video_path,
            human=True,
            ts=True,
            fb_config=config.fb_config,
            )
        env = wrappers.NormalizeActions(env)
    elif suite == 'calvingc':
        task = "_".join(task.split("_"))
        # Use CalvinGC (with RewardManager) for training, and simple CalvinPlayTable for eval/other modes.
        if mode == "train":
            reward_specs = _make_calvin_reward_specs(config)
            # breakpoint()
            env = wrappers.CalvinGC(
                dataset_path=config.dataset_path,
                task_name=task,
                reward_specs=reward_specs,
                reward_case=getattr(config, "reward_case", None),
                device=config.device,
                ep_len=config.time_limit,
                size=config.size,
                camera=config.camera,
                show_gui=getattr(config, "show_gui", False),
                scene=getattr(config, "scene", None),
            )
        else:
            env = wrappers.CalvinPlayTable(
                dataset_path=config.dataset_path,
                task_name=task,
                ep_len=config.time_limit,
                size=config.size,
                camera=config.camera,
                show_gui=getattr(config, "show_gui", False),
                scene=getattr(config, "scene", None),
            )
        env = wrappers.NormalizeActions(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    if config.resume_dir:
        logdir = pathlib.Path(config.resume_dir).expanduser()
        print(f"Resuming from existing directory: {logdir}")
    else:
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
    
    # print(wandb_device)
    match = re.search(r'dreamerr-(\d+)', socket.gethostname())
    if match:
        wandb_device = match.group(1) +':'+config.device[-1]
    else:
        wandb_device = socket.gethostname() +':'+config.device[-1]

    config.fb_config['fb_train_until']=config.fb_train_until
    
    '''
    wandb 
    '''
    config.experiment_name=str(logdir)
    run_dir = pathlib.Path("results") / config.project_name / config.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    '''
    wandb.require("core")
    wandb.init(config=config,
               project=config.project_name,
               entity=config.team_name,
               notes=wandb_device,
               name=config.experiment_name+"_"+str(config.seed),
               group=config.scenario_name,
               dir=str(run_dir),
               job_type="training",
               reinit=True)
    '''
    # breakpoint()
    # breakpoint()
    config.description = wandb_device +" "+ "prefill:"+str(config.prefill)+ " " +config.description
    # breakpoint()
    wandb.init(
        # config=config,
               project=config.project_name,
            #    workspace=config.team_name,
               notes=wandb_device,
               mode=config.wandb_mode,
               description=config.description,
               name=config.experiment_name,
               logdir=str(run_dir),
               reinit=True)
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    should_train_fb = tools.Until(config.fb_train_until)
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    
    suite,_ = config.task.split("_", 1)
    log_calvin_metrics = suite == 'calvingc'
    # breakpoint()
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
            suite=suite,
            log_calvin_metrics=log_calvin_metrics,
            save_frame_every=getattr(config, 'save_frame_every', 0),
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    reward_dataset = make_reward_dataset(train_eps, config)
    # breakpoint()
    agent = Dreamer(
        train_envs[0].obs_space,
        # train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        print(f"Loading checkpoint from {logdir / 'latest.pt'}")
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    def get_actual_env(env):
        """wrapper"""
        current = env
        for _ in range(10):
            if hasattr(current, 'setup_fb_encoder'):
                return current
            if hasattr(current, '_env'):
                current = current._env
            else:
                break
        return env

    print("Setting up FB encoder for all train_envs...")
    for i, env in enumerate(train_envs):
        actual_env = get_actual_env(env)
        if hasattr(actual_env, 'setup_fb_encoder'):
            print(f"[Train Env {i}] Found MetaWorldGC, setting up FB encoder...")
            actual_env.setup_fb_encoder(agent._wm.encoder, agent._wm.preprocess)
        else:
            print(f"[Train Env {i}] Warning: setup_fb_encoder not found!")
    print("FB encoder setup complete for all envs!")

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        is_train_fb = should_train_fb(agent._step)
        logger.write()
        if config.eval_episode_num > 0:
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
                suite=suite,
                log_calvin_metrics=log_calvin_metrics,
                save_frame_every=getattr(config, 'save_frame_every', 0),
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        # breakpoint()
        if log_calvin_metrics and config.reward_case in ("fb", "vaefb"):
            wandb.log(
                {"calvingc/train_fb_phase": float(is_train_fb), "step": logger.step},
                step=logger.step,
            )
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            save_dir=logdir,
            suite=suite,
            fb_train=reward_dataset if (config.reward_case == 'fb' or config.reward_case == 'vaefb') else None,
            is_train_fb=is_train_fb if (config.reward_case == 'fb' or config.reward_case == 'vaefb') else False,
            log_calvin_metrics=log_calvin_metrics,
            save_frame_every=getattr(config, 'save_frame_every', 0),
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "fb_network_dict": train_envs[0].get_fb_network() if hasattr(train_envs[0], "get_fb_network") and (config.reward_case == 'fb' or config.reward_case == 'vaefb') else None
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
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
    main(parser.parse_args(remaining))
    wandb.finish()