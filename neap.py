from argparse import ArgumentParser

from data_utils import DBP15K_PAIRS,DBP1M_PAIRS
from dbp15k import DBP15k
from dbp1m import DBP1M
from srprs import SRPRS, CPM_TYPES

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pair', type=str, default='en_fr')
    parser.add_argument('--use_fasttext', default=False, action='store_true')
    parser.add_argument('--use_cpm', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--init_type', type=str, default='MNEAP-L')
    parser.add_argument('--do_sinkhorn', default=False, action='store_true')

    args = parser.parse_args()
    with open("strategy.txt","w",encoding="utf-8") as f:
        f.write("SRS")
    
    pair = args.pair
    if pair in DBP15K_PAIRS:
        DBP15k('dataset/DBP15K', pair, 
              device=args.device,
              init_type=args.init_type,
              do_sinkhorn=args.do_sinkhorn)
    elif pair in DBP1M_PAIRS:
        DBP1M('dataset/DBP1M', pair, 
              device=args.device,
              init_type=args.init_type,
              do_sinkhorn=args.do_sinkhorn)
    else:
        SRPRS('dataset/SRPRS', pair,
              use_fasttext=args.use_fasttext,
              cpm_types=CPM_TYPES if args.use_cpm else None,
              device=args.device,
              init_type=args.init_type,
              do_sinkhorn=args.do_sinkhorn)
