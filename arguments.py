import argparse
# import deepspeed

def get_args():
    parser = argparse.ArgumentParser(description='iCarl2.0')
    
    parser.add_argument('--memory_size', type=int, default=int(1e5)) # 고정!
    parser.add_argument('--batch_size', type=int, default=128) # 고정!
    parser.add_argument('--gamma', type=float, default=0.99) # 고정!
    
    # PER parameters
    parser.add_argument('--alpha', type=float, default=0.2) # 고정!
    parser.add_argument('--beta', type=float, default=0.6) # 고정!
    parser.add_argument('--prior_eps', type=float, default=1e-6) # 고정!
    
    # Categorical DQN parameters
    parser.add_argument('--v_min', type=float, default=-10.0) # 고정!
    parser.add_argument('--v_max', type=float, default=10.0) # 고정!
    parser.add_argument('--atom_size', type=int, default=21) # 고정!
    
    # N-step Learning
    parser.add_argument('--n_step', type=int, default=3) # 고정!
    
    # Noisy Network
    parser.add_argument('--std', type=float, default=0.1)
    
    
    parser.add_argument('--lr', type=float, default=0.000125)
    
    parser.add_argument('--add_1_step_loss', type=bool, default=False)
    
    parser.add_argument('--no_tag', type=bool, default=False)
    
    args=parser.parse_args()

    return args