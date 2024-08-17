
import argparse
import torch
import hiera

parser = argparse.ArgumentParser(description='Calculate Flops')
parser.add_argument('--model_name', required=True, type=str, help='model name')
parser.add_argument('--package', default='fvcore', choices=['calflops', 'fvcore'], help='package to calculate flops')
args = parser.parse_args()

model = getattr(hiera, args.model_name)(pretrained=False, model_name=args.model_name)
print(args.model_name)
if args.package == 'calflops':
    from calflops import calculate_flops
    input_shape = (1, 3, 224, 224)
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("%s FLOPs:%s   MACs:%s   Params:%s \n" %(args.model_name, flops, macs, params))

elif args.package == 'fvcore':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    input_ = torch.randn(1, 3, 224, 224)
    flops = FlopCountAnalysis(model, input_)
    print(f'-------{args.model_name}----')
    print(flops.total())
    print(flop_count_table(flops))
    with open(f'flops/flops-{args.model_name}.txt', 'w') as f:
        f.write(flop_count_table(flops))
