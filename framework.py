from bootstrap import *
from eval import *
from graph_utils import *
from models import ModelWrapper
from utils import *
import torch
import torch.nn as nn


class FinalFantasy(nn.Module):
    def __init__(self, dataset, dim=200, name="",
                 device='cpu', srp=True, model='rrea', strategy="ASRS-TFIDF", 
                 fuse_semantic=True, fuse_lev_dist=True):
        super(FinalFantasy, self).__init__()
        ent_sizes = [dataset.ent1_len, dataset.ent2_len]
        self.cosine = not fuse_semantic
        self.lev = fuse_lev_dist
        rel_sizes = [dataset.rel1.max() + 1, dataset.rel2.max() + 1]
        self.ent_sizes = ent_sizes
        self.strategy=strategy
        self.name = name
        self.device = device
        self.srp = srp
        self.th=[1,1]
        # self.use_cache = use_cache
        pairs = SRPRS_PAIRS if srp else DBP15K_PAIRS
        for pair in pairs:
            if pair in self.name:
                self.pair = pair

        self.model_name = model
        self.edge_index = list(get_edge_index(dataset))
        self.edge_type = list(get_edge_type(dataset))
        self.graphs = [ind2sparse(self.edge_index[i], ent_sizes[i]) for i in range(2)]

        self.trusted = nn.ParameterList(
            [nn.Parameter(torch.zeros([sz]), requires_grad=False) for sz in ent_sizes]
        )
        self.trusted_mask = None
        self.dataset = dataset
        self.best_hits = {}
        self.last_hits = {}
        self.log_each_it = []

        # self.cached_sims = {}

        def clone_detach(t):
            return apply(lambda x: x.clone().detach(), *t)

        self.model = ModelWrapper(self.model_name,
                                  lang=self.pair,
                                  ei=clone_detach(self.edge_index),
                                  et=clone_detach(self.edge_type),
                                  link=self.dataset.total_y,
                                  srprs=self.srp,
                                  ent_sizes=ent_sizes,
                                  rel_sizes=rel_sizes,
                                  device=self.device,
                                  dim=dim)

    def get_trusted(self, which=None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if which is None:
            return self.get_trusted(0), self.get_trusted(1)
        return torch.masked_fill(self.trusted[which], self.trusted_mask[which], 0.)

    def confident_seed_generator(self, curr_it, conf, gt_sim):
        if isinstance(gt_sim, tuple):
            gt_sim_x, gt_sim_y = gt_sim
        else:
            gt_sim_x = matrix_argmax(gt_sim, dim=1)
            gt_sim_y = matrix_argmax(gt_sim, dim=0)
        device = gt_sim_x.device
        conf_mask = apply(lambda x: x > 0, *conf)
        gt_x = torch.stack(
            (torch.arange(gt_sim_x.size(0)).to(device, torch.long),
             gt_sim_x.squeeze()
             ), dim=0)
        ori_seeds=[(e1, e2) for e1, e2 in gt_x.t().cpu().numpy()]
        gt_y = torch.stack(
            (gt_sim_y.squeeze(),
             torch.arange(gt_sim_y.size(0)).to(device, torch.long),
             ), dim=0)
        gt_x, gt_y = gt_x[:, conf_mask[0]], gt_y[:, conf_mask[1]]
        seeds=[(e1, e2) for e1, e2 in gt_x.t().cpu().numpy()]
        set_x = set((e1, e2) for e1, e2 in gt_x.t().cpu().numpy())
        set_y = set((e1, e2) for e1, e2 in gt_y.t().cpu().numpy())
        if curr_it==0:
            self.px=set()
            self.py=set()
        for e1, e2 in set_y.intersection(set_x):
            self.px.add(e1)
            self.py.add(e2)
        # gt = np.array(list(self.set_xy)).T
        gt = np.array(list(set_y.intersection(set_x))).T
        ground_truth = getattr(self.dataset, 'total_y').tolist()
        ground_truth = set((ground_truth[0][i],ground_truth[1][i]) for i in range(len(ground_truth[0])))
        #seeds=list(set_y.intersection(set_x))
        #seeds=[(e1, e2) for e1, e2 in gt_x.t().cpu().numpy()]
        
        # truth_gmd0,false_gmd0,truth_gmd1,false_gmd1,truth_gmd2,false_gmd2=0,0,0,0,0,0
        # truth_num0,false_num0,truth_num1,false_num1,truth_num2,false_num2=0,0,0,0,0,0
        # for i in range(len(ori_seeds)):
            # if ori_seeds[i] in ground_truth:
                # truth_gmd0+=self.gmd[i].item()
                # truth_num0+=1
            # else:
                # false_gmd0+=self.gmd[i].item()
                # false_num0+=1
            # if ori_seeds[i] not in seeds:
                # if ori_seeds[i] in ground_truth:
                    # truth_gmd1+=self.gmd[i].item()
                    # truth_num1+=1
                # else:
                    # false_gmd1+=self.gmd[i].item()
                    # false_num1+=1
            # else:
                # if ori_seeds[i] in ground_truth:
                    # truth_gmd2+=self.gmd[i].item()
                    # truth_num2+=1
                # else:
                    # false_gmd2+=self.gmd[i].item()
                    # false_num2+=1
        # with open("th_analysis.txt","a+") as f:
            # f.write("candidate_seeds=%d,th=%f,truth_avg_gmd=%f,truth_num=%d,false_avg_gmd=%f,false_num=%d\n"%(len(ori_seeds),self.th[1].item(),truth_gmd0/truth_num0,truth_num0,false_gmd0/false_num0,false_num0))
            # f.write("unselected_seeds=%d,th=%f,truth_avg_gmd=%f,truth_num=%d,false_avg_gmd=%f,false_num=%d\n"%(len(ori_seeds)-len(seeds),self.th[1].item(),truth_gmd1/truth_num1,truth_num1,false_gmd1/false_num1,false_num1))
            # f.write("seeds=%d,th=%f,truth_avg_gmd=%f,truth_num=%d,false_avg_gmd=%f,false_num=%d\n\n"%(len(seeds),self.th[1].item(),truth_gmd2/truth_num2,truth_num2,false_gmd2/false_num2,false_num2))
        print("gmd th:",self.th[0].item())
        # print("truth_avg_gmd:",truth_gmd/truth_num)
        # print("truth_num:",truth_num)
        # print("false_avg_gmd:",false_gmd/false_num)
        # print("false_num:",false_num)
        #assert(1==2)
        return gt

    def get_gnn_embed(self) -> List[Tensor]:
        embeds = self.model.get_curr_embeddings()
        embeds = apply(lambda x: x.to(self.device), *embeds)
        return embeds

    def get_similarity_matrix(self, embed=None, which=0):
        if embed is None:
            embed = self.get_gnn_embed()
        if which == 1:
            embed = reversed(embed)
        return cosine_sim(*embed)

    def update_mask(self, x2y_argmax, y2x_argmax, lens):
        #print(x2y_argmax, y2x_argmax, lens)
        if x2y_argmax.dim() == 2:
            x2y_argmax, y2x_argmax = apply(lambda x: x[:, :1].squeeze(), x2y_argmax, y2x_argmax)
        masks = get_bi_mapping(x2y_argmax, y2x_argmax, lens)
        self.trusted_mask = apply(torch.logical_not, *masks)

    @torch.no_grad()
    def reconstruct_adj(self,num_it=20,curr_it=0):
        embed = self.get_gnn_embed()
        size = apply(lambda x: x.size(0), *embed)
        x2y_val, x2y_argmax = self.get_fused_sim('x2y', embed).topk(dim=1, k=2)
        y2x_val, y2x_argmax = self.get_fused_sim('y2x', embed).topk(dim=1, k=2)
        graph = iterative_completion(x2y_argmax, y2x_argmax, self.edge_index, size, (x2y_val, y2x_val), strategy=self.strategy)
        self.graphs = [rebuild_with_indices(graph[i] + self.graphs[i], strategy=self.strategy) for i in range(2)]
        self.edge_index = [g._indices() for g in self.graphs]

    @torch.no_grad()
    def get_fused_sim(self, x2y='x2y', embed=None, use_embed=True):
        if use_embed:
            if embed is None:
                embed = self.get_gnn_embed()
            if x2y == 'y2x':
                embed = reversed(embed)
        return self.fuse_sims(cosine_sim(*embed) if use_embed else 0, x2y)

    @torch.no_grad()
    def fuse_sims(self, M, x2y='x2y', fuse_params=None, weight=(0.2, 0.2)):
        if fuse_params is None:
            fuse_params = ('cosine' if self.cosine else 'sim', 'lev')
            if not self.lev:
                fuse_params = fuse_params[:-1]
                weight = weight[:-1]
        for i, param in enumerate(fuse_params):
            M += self.get_initial_m(param, x2y, to_tensor=True, device=self.device) * weight[i]
        return M

    @torch.no_grad()
    def get_initial_m(self, ty, x2y, to_tensor=True, device='cpu', minmax=False):
        # sim_name = '_'.join(['sim', ty, x2y])
        # file_pos = '{0}_vectors/{1}.pt_{2}_{3}.npy'.format(['dbp', 'srprs'][self.srp], self.pair, ty, x2y)

        # if self.use_cache and sim_name in self.cached_sims:
        #     x = self.cached_sims[sim_name]
        # else:
        #     x = np.load(file_pos)
        #     if self.use_cache:
        #         self.cached_sims[sim_name] = x
        # if to_tensor:
        #     x = torch.from_numpy(x).to(device)
        x = to_dense(getattr(self.dataset, '_'.join([ty, x2y])).to(device))
        if minmax:
            x = masked_minmax(x)
        return x

    @torch.no_grad()
    def get_current_perm(self, is_first=False, to_argmax=False, fuse_csls=True):
        embed = None if is_first else self.get_gnn_embed()
        P = bi_csls_matrix(self.get_fused_sim('x2y', embed, not is_first),
                           self.get_fused_sim('y2x', embed, not is_first))
        if is_first:
            evaluate_sim_matrix(self.dataset.total_y, *P)
        if to_argmax:
            P = P[0] + P[1].t()
            return matrix_argmax(P, 1), matrix_argmax(P, 0)
        return P[0] + P[1].t() if fuse_csls else P

    @torch.no_grad()
    def update_score(self, num_it, curr_it, perm, which=None):
        if which is None:
            self.update_score(num_it, curr_it, perm, 0)
            self.update_score(num_it, curr_it, perm, 1)
            return
        if isinstance(perm, tuple):
            perm = perm[which]
        elif which == 1:
            perm = perm.t()
        dist = graph_matching_distance(self.graphs[which],
                                       self.graphs[1 - which],
                                       perm,
                                       argmax=perm.dim() == 2)
 
        if curr_it==0:
            mean_dist=dist.mean()
            self.th[which]=mean_dist
        else:
            if self.strategy=="ASRS-TFIDF" or self.strategy=="ASRS-PageRank":
                #first_mean
                mean_dist=self.th[which]
            else:
                #curr_mean
                mean_dist = dist.mean()
                self.th[which]=mean_dist
        # #th=(th+mean)/2
        # if curr_it==0:
            # mean_dist=dist.mean()
            # self.th[which]=mean_dist
        # else:
            # mean_dist=(self.th[which]+dist.mean())/2
            # self.th[which]=mean_dist
        #mean_dist = min(dist.mean()*0+1.0,dist.mean()+(curr_it/num_it))
        #mean_dist = dist.mean()
        #mean_dist = dist.mean()*0+1.0
        print("GMD max:",dist.max().item(),"GMD min:", dist.min().item())
        print("GMD th:",dist.mean().item(),"->", mean_dist.item())
        tidx, eidx = dist < mean_dist, dist > mean_dist
        if which==1:
            self.gmd=dist
        self.trusted[1 - which][tidx] = 1
        self.trusted[1 - which][eidx] = 0

    def th_mwgm_iteration(self, iter_type, sim_th=0.7, k=10):
        # TH=0.7, K=10 from OpenEA
        sim_mat = cosine_sim(*self.model.get_curr_embeddings(self.device)).cpu().numpy()
        if iter_type == 'th':
            seeds = find_potential_alignment_greedily(sim_mat, sim_th)  # according to OpenEA
        elif iter_type == 'mwgm':
            seeds = find_potential_alignment_mwgm(sim_mat, sim_th, k)
        else:
            raise NotImplementedError()
        return np.array(list(seeds)).T

    @torch.no_grad()
    def dat_iteration(self, alpha=0.05):
        # alpha=0.05 from DAT
        embed = self.model.get_curr_embeddings(self.device)
        sim_mat = cosine_sim(*embed)
        x2y_val, x2y_argmax = sim_mat.topk(dim=1, k=2)
        y2x_val, y2x_argmax = sim_mat.t().topk(dim=1, k=2)
        size = apply(lambda x: x.size(0), *embed)
        seeds = filter_mapping(x2y_argmax, y2x_argmax, size, (x2y_val, y2x_val), alpha)
        return seeds.cpu().numpy()
    
    @torch.no_grad()
    def new_iteration(self, num_it, curr_it, alpha=0):
        import networkx as nx
        from collections import defaultdict
        from text_utils import get_tf_idf
        embed = self.model.get_curr_embeddings(self.device)
        sim_mat = cosine_sim(*embed)
        #print(type(sim_mat))
        #print(self.graphs)
        indices1,indices2=self.edge_index[0],self.edge_index[1]
        
        words1,words2=[str(i) for i in range(self.ent_sizes[0])],[str(i) for i in range(self.ent_sizes[1])]
        name_words1,name_words2=["" for i in range(self.ent_sizes[0])],["" for i in range(self.ent_sizes[1])]
        
        edges1,edges2=defaultdict(set),defaultdict(set)
        for i in range(len(indices1[0])):
            a,b=int(indices1[0][i].item()),int(indices1[1][i].item())
            edges1[a].add(b)
            edges1[b].add(a)
        for i in range(len(indices2[0])):
            a,b=int(indices2[0][i].item()),int(indices2[1][i].item())
            edges2[a].add(b)
            edges2[b].add(a)
        for a,b in edges1.items():
            name_words1[a]+=" ".join([str(it) for it in b])
        for a,b in edges2.items():
            name_words2[a]+=" ".join([str(it) for it in b])
        #print(words1[:5])
        #print(name_words1[:5])
        tfidf1,tfidf2=get_tf_idf(words1, name_words1, None),get_tf_idf(words2, name_words2, None)
        tfidf1,tfidf2=to_torch_sparse(tfidf1, device=self.device).to(float).to_dense(),to_torch_sparse(tfidf2, device=self.device).to(float).to_dense()
        # # sim_x2y=remain_topk_sim(sim_mat,k=10).to(float)
        # # tgm=spspmm(tfidf1,sim_x2y)
        # # sim_x2y=spspmm(tgm, tfidf2).to_dense()
        # #tgm=tfidf1.mm(sim_mat.to(float))
        # #sim_x2y=tgm.mm(tfidf2)
        # sim_x2y=remain_topk_sim(tfidf1.mm(sim_mat.to(float)),k=3).to(float)
        # indices_x2y=sim_x2y._indices().cpu().numpy()
        # set_x = set((indices_x2y[0][i], indices_x2y[1][i]) for i in range(len(indices_x2y[0])))
        # #tgm=tfidf2.mm(sim_mat.t().to(float))
        # #sim_y2x=tgm.mm(tfidf1)
        # sim_y2x=remain_topk_sim(tfidf2.mm(sim_mat.t().to(float)),k=3).to(float)
        # indices_y2x=sim_y2x._indices().cpu().numpy()
        # set_y = set((indices_y2x[1][i], indices_y2x[0][i]) for i in range(len(indices_y2x[0])))
        # seeds = set_y.intersection(set_x)
        
        x2y_val, x2y_argmax = (0.5*(tfidf1.mm(remain_topk_sim(sim_mat.to(float), k=1).to_dense()).mm(tfidf2))+0.5*sim_mat.to(float)).topk(dim=1, k=10000)
        y2x_val, y2x_argmax = (0.5*(tfidf2.mm(remain_topk_sim(sim_mat.t().to(float), k=1).to_dense()).mm(tfidf1))+0.5*sim_mat.t().to(float)).topk(dim=1, k=10000)
        size = apply(lambda x: x.size(0), *embed)
        seeds = filter_mapping(x2y_argmax, y2x_argmax, size, (x2y_val, y2x_val), alpha)
        print("seeds(%d)"%(seeds.shape[1]))
        return seeds.cpu().numpy()
        
        print("seeds(%d):"%(len(seeds)),list(seeds)[:5])
        return np.array(list(seeds)).T
        
        # G1,G2 = nx.DiGraph(),nx.DiGraph()
        # G1.add_nodes_from(list(edges1.keys()))
        # G2.add_nodes_from(list(edges2.keys()))
        # for i in range(len(indices1[0])):
            # a,b=int(indices1[0][i].item()),int(indices1[1][i].item())
            # G1.add_edge(a, b)
            # G1.add_edge(b, a)
        # for i in range(len(indices2[0])):
            # a,b=int(indices2[0][i].item()),int(indices2[1][i].item())
            # G2.add_edge(a, b)
            # G2.add_edge(b, a)
        # pr1,pr2=nx.pagerank(G1,alpha=0.85),nx.pagerank(G2,alpha=0.85)
        # it1,it2=sorted(pr1.items(),key=lambda x:x[1],reverse=True),sorted(pr2.items(),key=lambda x:x[1],reverse=True)
        # seeds=list()
        # y_flag=defaultdict(int)
        # for i in range(8000):
            # candidate_xy=list()
            # for j in it2.keys():
                # if y_flag[j]==0:
                    # candidate_xy.append(((i,j),sim_mat[it1[i][0],it2[j][0]]))
            # candidate_xy=sorted(candidate_xy,key=lambda x:x[1],reverse=True)
            # seeds.append(candidate_xy[0])

    @torch.no_grad()
    def refinement(self, num_it, curr_it, refine_begin, iter_type):
        assert iter_type in ['ours', 'mraea', 'dat', 'th', 'mwgm', 'new', 'none']
        if curr_it >= refine_begin:
            if iter_type == "ours":
                self.reconstruct_adj(num_it=20,curr_it=curr_it)
            elif iter_type == "mraea":
                self.model.mraea_iteration()
            elif iter_type == 'dat':
                self.model.update_trainset(self.dat_iteration(), append=True)
            elif iter_type == 'new':
                self.reconstruct_adj(num_it=20,curr_it=curr_it)
                self.model.update_trainset(self.new_iteration(num_it, curr_it), append=True)
            elif iter_type == 'none':
                pass
            else:
                self.model.update_trainset(self.th_mwgm_iteration(iter_type), append=True)
        is_first = curr_it == 0
        if iter_type == 'ours' or is_first:
            perm: Tuple[Tensor, Tensor] = self.get_current_perm(is_first, to_argmax=True)
            if is_first:
                self.update_mask(*perm, lens=self.ent_sizes)
            self.update_score(num_it, curr_it, perm)
            self.model.update_trainset(self.confident_seed_generator(curr_it, self.get_trusted(), gt_sim=perm),append=False)

    def train_myself(self, num_it=20, epoch=20, refine_begin=1, iter_type='ours', only_gnn=False):
        assert refine_begin <= num_it
        for curr_it in range(num_it):
            print("Iteration {0}".format(curr_it))
            self.refinement(num_it, curr_it, refine_begin, iter_type)
            self.model.train1step(epoch)
            self.eval_myself(only_gnn)

    def update_best_hits(self, hits, task):
        if isinstance(hits, tuple):
            hits = hits[0]
        best_hits = self.best_hits.get(task, {})
        for k, v in hits.items():
            if best_hits.get(k, 0.) < v:
                best_hits[k] = v
        self.last_hits[task] = hits
        self.best_hits[task] = best_hits
        # print(self.best_hits)

    def new_log(self, **kwargs):
        self.log_each_it.append(kwargs)
        it = len(self.log_each_it)
        name = self.name
        np.save("log/{0}_{1}".format(name, it), self.get_current_perm().cpu().numpy())

    @torch.no_grad()
    def eval_myself(self, only_gnn=False, eval_tasks=None):
        if eval_tasks is None:
            # eval_tasks = ["total_y", "test_y", "hard_y", "val_y", "hard_val_y"]
            eval_tasks = ['total_y']
        self.eval()
        print("-----------------------EVAL-----------------------")
        curr_log = {}
        for task in eval_tasks:
            if hasattr(self.dataset, task):
                print('Eval task is', task)
                ground_truth = getattr(self.dataset, task)
                if not only_gnn:
                    perm = self.get_current_perm()
                    result = evaluate_sim_matrix(ground_truth, perm)
                    self.update_best_hits(result, task)
                    curr_log[task + "_final"] = result
                    perm0 = self.get_fused_sim('x2y')
                    perm1 = self.get_fused_sim('y2x')
                    print("FUSE")
                    curr_log[task + "_fuse"] = evaluate_sim_matrix(ground_truth, perm0, perm1, no_csls=False)

                perm0 = self.get_similarity_matrix(which=0)
                perm1 = self.get_similarity_matrix(which=1)
                print("gnn hitk")
                curr_log[task + "_structure"] = evaluate_sim_matrix(ground_truth, perm0, perm1, start="\t")

        curr_log['seed'] = self.model.test_train_pair_acc()
        self.new_log(**curr_log)
        self.train()
