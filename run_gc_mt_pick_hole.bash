seed=$3
steps=1000001
logdir='./logdir'
eval_episode_num=10
task=metaworldgc_pick-out-of-hole
bash_debug=false
wandb_mode=cloud
debug_flag="$1"
run_file='goal_dreamer.py'
time_limit=256
video_path="./videos/seed1_val_0_pick_up_the_blue_fork_Trossen_WidowX_250_robot_arm_0.mp4"
video_name=$(basename "$video_path")

reward_type='dense'
sparse_interval=64

crop=false

alpha=0.01
beta_scale=0.00001
reward_case='vaefb'
fb_train_until=100000

echo -e "\033[33mreward_case: $reward_case\033[0m"

interval_part="interval 1"
if [ "$reward_type" = "sparse" ]; then
    interval_part="interval ${sparse_interval}"
fi

description="${reward_type} ${reward_case} reward，alpha${alpha}，beta${beta_scale}，time_limit ${time_limit}，${interval_part}, fb_train ${fb_train_until}, crop ${crop}，video: ${video_name}"

echo -e "\033[33mdescription: $description\033[0m"

if [ -z "$2" ]; then
    device="cuda:0"
else
    device="cuda:$2"
fi

if [ "$debug_flag" == "1" ]; then
    wandb_mode=local
    bash_debug=true
    eval_episode_num=1
    steps=1500
fi

echo -e "\033[33mdevice: $device\033[0m"
echo -e "\033[33mbash_debug: $bash_debug\033[0m"
echo -e "\033[0mdebug_flag: $debug_flag\033[0m"
echo -e "\033[33mreward_type: $reward_type\033[0m"
echo -e "\033[33malpha: $alpha\033[0m"

echo -e "\033[33mtask: $task\033[0m"
echo -e "\033[33mseed: $seed\033[0m"

if [ "$bash_debug" == "true" ]; then
    now_config='defaults debug_metaworld'
    is_debug=True
else
    now_config='defaults metaworld'
    is_debug=False
fi

python3 $run_file --device $device \
--task $task \
--seed $seed \
--steps $steps \
--logdir $logdir \
--is_debug $is_debug \
--description "$description" \
--time_limit $time_limit \
--wandb_mode $wandb_mode \
--eval_episode_num $eval_episode_num \
--project_name "goal_dream" \
--video_path "$video_path" \
--reward_type "$reward_type" \
--reward_case "$reward_case" \
--crop $crop \
--alpha $alpha \
--beta_scale $beta_scale \
--fb_train_until $fb_train_until \
--sparse_interval $sparse_interval \
--configs $now_config
