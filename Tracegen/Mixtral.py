from function import System

class ModelMixtral(System):
    def __init__(self, model_args, args):
        super().__init__(args)
        self.dim = model_args.dim
        self.dim_expert = model_args.dim_expert
        self.n_expert = model_args.n_expert
        self.top_k = model_args.top_k

    def memory_mapping(self):
        self.w_up_proj_bo = []
        
