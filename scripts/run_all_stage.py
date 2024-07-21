import os
import argparse
import glob
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
        "--raw",
        action="store_true"
    )
    parser.add_argument(
        "--sd",
        action="store_true"
    )
    parser.add_argument(
        "--mesh",
        action="store_true"
    )
    args, extras = parser.parse_known_args()
    
    prompt: str = args.prompt
    prompt_rm: str = prompt.replace(' ', "_")
    seed: int = args.seed
    gpu: int = args.gpu
    tag = prompt_rm+"-"+str(seed)
    stage1_cfg = "sd-stage1" if args.sd else ("mvdream-stage1" if args.raw else "mvdream-shading-stage1")
    
    dir_stage1 = "cfd-sd-stage1" if args.sd else ("cfd-mvdream-stage1" if args.raw else "cfd-mvdream-shading-stage1")
    dir_mesh_stage2 = "cfd-mesh-geometry-stage2"
    dir_mesh_stage3 = "cfd-mesh-texture-stage3"
    dir_nerf_stage2 = "cfd-nerf-stage2"
    
    # stage 1
    cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/{stage1_cfg}.yaml name={dir_stage1} tag=\"{tag}\" system.prompt_processor.prompt=\"{prompt}\" seed={seed}'
    print()
    print(cmd)
    ret = os.system(cmd)
    if ret!=0:
        raise RuntimeError
    
    if args.mesh:
        # stage 2
        folder_fn = sorted(glob.glob(f'{tag}@*', root_dir=f'outputs/{dir_stage1}'))[-1]
        geometry_convert_from = os.path.join("outputs", dir_stage1, folder_fn, "ckpts", "last.ckpt")
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-mesh-geometry-stage2.yaml name={dir_mesh_stage2} tag=\"{tag}\" system.geometry_convert_from=\"{geometry_convert_from}\" system.prompt_processor.prompt=\"{prompt}\" seed={seed}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret!=0:
            raise RuntimeError
        
        # stage 3
        folder_fn = sorted(glob.glob(f'{tag}@*', root_dir=f'outputs/{dir_mesh_stage2}'))[-1]
        geometry_convert_from = os.path.join("outputs", dir_mesh_stage2, folder_fn, "ckpts", "last.ckpt")
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-mesh-texture-stage3.yaml name={dir_mesh_stage3} tag=\"{tag}\" system.geometry_convert_from=\"{geometry_convert_from}\" system.prompt_processor.prompt=\"{prompt}\" seed={seed}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret!=0:
            raise RuntimeError
    else:
        # stage 2
        folder_fn = sorted(glob.glob(f'{tag}@*', root_dir=f'outputs/{dir_stage1}'))[-1]
        geometry_convert_from = os.path.join("outputs", dir_stage1, folder_fn, "ckpts", "last.ckpt")
        cmd = f'python launch.py --train --gpu {gpu} --config configs/cfd/sd-nerf-stage2.yaml name={dir_nerf_stage2} tag=\"{tag}\" system.geometry_convert_from=\"{geometry_convert_from}\" system.prompt_processor.prompt=\"{prompt}\" seed={seed}'
        print()
        print(cmd)
        ret = os.system(cmd)
        if ret!=0:
            raise RuntimeError
