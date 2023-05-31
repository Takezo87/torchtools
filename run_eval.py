from torchtools.eval_utils import *
import argparse



if __name__ == '__main__':
    # pool=Pool(2)
    parser = argparse.ArgumentParser(description='parse eval options')

    parser.add_argument("--path_base", help="full path of the base dataframe")
    parser.add_argument("--fn_result", help="filename of the result dataframe")
    # parser.add_argument("-S", "--season", help="specify season to select content")
    # parser.add_argument("-N", "--new", default=False, action='store_true')
    parser.add_argument("--is_colab", default=False, action='store_true')
    parser.add_argument("--is_class", default=False, action='store_true')
    parser.add_argument("--opp", default=False, action='store_true')
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--quantile", type=float, default=.9)
    parser.add_argument("--end_train", type=int, default=170000)
    parser.add_argument("--end_valid", type=int, default=230000)
    parser.add_argument("--end_test", type=int, default=230005)
    parser.add_argument("--idxs" ,type=int, action='append')
    # parser.add_argument("-U", "--update", default=None)
    # parser.add_argument("--sports", type=str, default='Soccer')
    parser_args, _ = parser.parse_known_args(sys.argv) #known args and unknown
    args = vars(parser_args)
    print (args)
    end_train, end_valid, end_test = args['end_train'], args['end_valid'], args['end_test']


    eval_config = EvalConfig(args['is_colab'], args['is_class'], args['path_base'], fn=args['fn_result'],
            preprocess=False)
    print(eval_config.df_results.shape)
    print(eval_config.base_dir)
    print(eval_config.df_base.shape)
    # if args['opp']:
    eval_config._df_base = eval_config._df_base.loc[~eval_config._df_base['pl_ah'].isna()]
    print(eval_config.df_base.shape)

    if args['opp']:
        idxs_r = basic_eval_ou(eval_config, args['idxs'], end_train=end_train, end_val=end_valid,
                pl_cols=['pl_ah', 'pl_ah_opp'], quantile=args['quantile'])
        # print(bas_eval)
        idxs_c = basic_eval_ou(eval_config, args['idxs'], end_train=end_train, end_val=end_valid,
                pl_cols=['pl_ah', 'pl_ah_opp'], quantile=args['quantile'], complement=True)
        # print(bas_eval)

        idxs_union = combine_idxs(idxs_r, idxs_c)
        idxs_intersection = combine_idxs_2(idxs_r, idxs_c)
        print(len(idxs_union), len(idxs_intersection))
        print('regular')
        print(eval_config.df_base.iloc[end_train:end_valid].iloc[idxs_r][['pl_ah', 'pl_1x2']].agg(['mean', 'count', 'sum']))
        print('complement')
        print(eval_config.df_base.iloc[end_train:end_valid].iloc[idxs_c][['pl_ah_opp', 'pl_1x2_opp']].agg(['mean', 'count', 'sum']))
    else:
        q = args['quantile']
        test = args['test']
        target_cols = ['pl_ah', 'pl_1x2']
        idxs_r = basic_eval(eval_config,args['idxs'], quantile=q, complement=False, test=test,
                target_cols=target_cols, end_train=end_train, end_valid=end_valid, end_test=end_test)
        idxs_c = basic_eval(eval_config,args['idxs'], quantile=q, complement=True, test=test,
                target_cols=target_cols, end_train=end_train, end_valid=end_valid, end_test=end_test)
        idxs_union = combine_idxs(idxs_r, idxs_c)
        idxs_intersection = combine_idxs_2(idxs_r, idxs_c)
        df_base = eval_config.df_base

        splits = list(range(end_train, end_valid)) if not test else list(range(end_valid, end_test))
        print(df_base.iloc[splits].iloc[idxs_r][target_cols].agg(['mean', 'sum', 'count']))
        print(df_base.iloc[splits].iloc[idxs_c][target_cols].agg(['mean', 'sum', 'count']))
        print(df_base.iloc[splits].iloc[idxs_intersection][target_cols].agg(['mean', 'sum', 'count']))
        print(df_base.iloc[splits].iloc[idxs_union][target_cols].agg(['mean', 'sum', 'count']))
    # print('interseciont')
    # print(eval_config.df_base.iloc[end_train:end_valid].iloc[idxs_intersection][['pl_ah', 'pl_1x2']].agg(['mean', 'count', 'sum']))
    # print('union')
    # print(eval_config.df_base.iloc[end_train:end_valid].iloc[idxs_union][['pl_ah', 'pl_1x2']].agg(['mean', 'count', 'sum
