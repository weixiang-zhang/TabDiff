import os
import fire


class Runner(object):
    def __init__(self):
        self.datasets = ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news']
    
    def _execute(self, args, cuda=0, file="main"):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        use_cuda = f"CUDA_VISIBLE_DEVICES={str(cuda)} "
        script = use_cuda + f"python {file}.py " + args
        print(f"Running: {script}")
        os.system(script)
    
    @staticmethod
    def _arg_dict_to_str(args_dic):
        STORE_TRUE = None
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not STORE_TRUE:
                args_list.append(str(val))
        Runner._print_args_list(args_list)
        return " ".join(args_list)
    
    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def train(self, cuda):
        dataset = "adult"
        args = f"--dataname {dataset} --mode train --deterministic --no_wandb"
        self._execute(args, cuda)

    def eval(self, cuda):
        dataset = "default"
        args = f"--dataname {dataset} --mode test --report --no_wandb"
        self._execute(args, cuda)

    def eval_syn(self, cuda=0):
        # conda activate synthcity (torch2.6.0 is ok)...
        dataset = "default"
        args = f"--dataname {dataset}"
        self._execute(args, cuda, file="eval/eval_quality")




if __name__ == "__main__":
    runner = Runner()
    fire.Fire(runner)
    ## python run.py train 0