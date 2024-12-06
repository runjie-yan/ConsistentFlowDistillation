import argparse
import glob
import os

# python scripts/run_shading_teaser.py --gpu xxx --prompt "yyy"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--raw-prompt",
        type=str,
        default=None,
    )
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--sd", action="store_true")
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_perp_neg", action="store_true")
    parser.add_argument("--skip_coarse", action="store_true")
    args, extras = parser.parse_known_args()

    prompt: str = args.prompt
    raw_prompt: str = args.raw_prompt if args.raw_prompt is not None else prompt
    prompt_rm: str = prompt.replace(" ", "_")
    raw_prompt_rm: str = raw_prompt.replace(" ", "_")
    
    seed: int = args.seed
    gpu: int = args.gpu
    enable_wandb: bool = "true" if args.wandb else "false"
    use_perp_neg: bool = "true" if args.use_perp_neg else "false"
    tag = prompt_rm + "-" + str(seed)
    raw_tag = raw_prompt_rm + "-" + str(seed)
    stage1_cfg = (
        "sd-stage1"
        if args.sd
        else ("mvdream-stage1" if args.raw else "mvdream-shading-stage1")
    )

    dir_stage1 = (
        "cfd-sd-stage1"
        if args.sd
        else ("cfd-mvdream-stage1" if args.raw else "cfd-mvdream-shading-stage1")
    )
    dir_mesh_stage2 = "cfd-mesh-geometry-stage2"
    dir_mesh_stage3 = "cfd-mesh-texture-stage3"
    dir_nerf_stage2 = f"cfd-nerf-stage2{args.tag}"

    # stage 1
    if args.skip_coarse:
        print("Skipping stage 1")
    else:
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/{stage1_cfg}.yaml name={dir_stage1} tag="{raw_tag}" system.prompt_processor.prompt="{raw_prompt}" seed={seed} system.loggers.wandb.enable={enable_wandb}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError

    if args.mesh:
        # stage 2
        folder_fn = sorted(glob.glob(f"{raw_tag}@*", root_dir=f"outputs/{dir_stage1}"))[-1]
        geometry_convert_from = os.path.join(
            "outputs", dir_stage1, folder_fn, "ckpts", "last.ckpt"
        )
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-mesh-geometry-stage2.yaml name={dir_mesh_stage2} tag="{tag}" system.geometry_convert_from="{geometry_convert_from}" system.prompt_processor.prompt="{prompt}" seed={seed} system.loggers.wandb.enable={enable_wandb} system.prompt_processor.use_perp_neg={use_perp_neg}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError

        # stage 3
        folder_fn = sorted(
            glob.glob(f"{tag}@*", root_dir=f"outputs/{dir_mesh_stage2}")
        )[-1]
        geometry_convert_from = os.path.join(
            "outputs", dir_mesh_stage2, folder_fn, "ckpts", "last.ckpt"
        )
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-mesh-texture-stage3.yaml name={dir_mesh_stage3} tag="{tag}" system.geometry_convert_from="{geometry_convert_from}" system.prompt_processor.prompt="{prompt}" seed={seed} system.loggers.wandb.enable={enable_wandb} system.prompt_processor.use_perp_neg={use_perp_neg}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError
    else:
        # stage 2
        folder_fn = sorted(glob.glob(f"{raw_tag}@*", root_dir=f"outputs/{dir_stage1}"))[-1]
        geometry_convert_from = os.path.join(
            "outputs", dir_stage1, folder_fn, "ckpts", "last.ckpt"
        )
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-nerf-stage2.yaml name={dir_nerf_stage2} tag="{tag}" system.geometry_convert_from="{geometry_convert_from}" system.prompt_processor.prompt="{prompt}" seed={seed} system.loggers.wandb.enable={enable_wandb} system.prompt_processor.use_perp_neg={use_perp_neg}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError
