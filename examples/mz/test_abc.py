import warnings
import os

import torch
from tensordict import TensorDict
import ray
from verl import DataProto
from verl.single_controller import Worker
from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import parallel_put


warnings.filterwarnings("ignore")

ray.init()


@ray.remote
class GPUWorker(Worker):

    def __init__(self):
        super().__init__()

    def env_info(self):
        return (
            f"[{self.get_name()}] "
            f"rank: {self.rank}, "
            f"world_size: {self.world_size}, "
            f"local_world_size: {os.environ.get('LOCAL_WORLD_SIZE')}, "
            f"local_rank: {os.environ.get('LOCAL_RANK')}, "
            f"master_addr: {os.environ.get('MASTER_ADDR')}, "
            f"master_port: {os.environ.get('MASTER_PORT')}, "
            f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    def get_name(self):
        ctx = ray.runtime_context.get_runtime_context()
        return ctx.get_actor_name()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.ALL)
    def add(self, x, y):
        return f"[{self.get_name()}] {x + y}"

    @register(dispatch_mode=Dispatch.DP_COMPUTE, execute_mode=Execute.ALL, blocking=False)
    def dummy_compute(self, data):
        for key in data.batch.keys():
            data.batch[key] += self.rank
        return data


if __name__ == "__main__":
    bsz = 4096
    seqlen = 512
    data_dict = {
        "x1": torch.randint(0, 10000, (bsz, seqlen)),
        "x2": torch.randint(0, 10000, (bsz, seqlen)),
    }
    data = DataProto.from_dict(tensors=data_dict)

    resource_pool = RayResourcePool([2], use_gpu=True, max_colocate_count=1)
    class_with_args = RayClassWithInitArgs(cls=GPUWorker)
    worker_group = RayWorkerGroup(resource_pool, class_with_args)
    worker_names = worker_group.worker_names
    workers = worker_group.workers

    print(ray.get([worker.env_info.remote() for worker in workers]))

    print(worker_group.add(x=1, y=2))

    data_list = data.chunk(worker_group.world_size)
    data_list_ref = parallel_put(data_list)
    print(f"data_ref type: {type(data_list_ref[0])}")

    outptu_ref = worker_group.dummy_compute(data_list_ref)
    output_lst = ray.get(outptu_ref)

    ray.shutdown()
