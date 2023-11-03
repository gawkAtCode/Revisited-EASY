from dists import *
from utils import *

def get_character_tf_idf(words, ent_lists):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    t = lambda x: list(enumerate(list(x)))
    vectorizer = CountVectorizer(vocabulary=words, lowercase=False, tokenizer=t, binary=False)
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(ent_lists)
    tfidf = transformer.fit_transform(X)
    return tfidf

def character_level_similarity(src_w2e: Tensor, trg_w2e: Tensor, src_word_x: Tensor, trg_word_x: Tensor, C2C: Tensor, sparse_k=1,
                           dense_mm=False, do_sinkhorn=True):
    # sim: Tensor = cosine_sim(src_word_x, trg_word_x)
    sim = cosine_sim(src_word_x, trg_word_x)
    # t1=time.clock()
    # sim = sinkhorn_process(sim)
    # t2=time.clock()
    # print("sinkhorn time: %d(min)"%(int((t2-t1)/60)))
    if sparse_k is None:
        # print(src_w2e.size(), sim.size(), trg_w2e.size())
        tgm = spmm(src_w2e.t(), sim)
        tgm = spmm(trg_w2e.t(), tgm.t()).t()
    else:
        # sim_val, sim_id = torch.topk(sim, sparse_k)
        # id_x = torch.arange(src_word_x.size(0), dtype=torch.long).to(sim_id.device).view(-1, 1).expand_as(sim_id)
        # ind = torch.stack([id_x.view(-1), sim_id.view(-1)], dim=0)
        # sim = ind2sparse(ind, sim.size(), values=sim_val.view(-1)).to(float)
        # # src_w2e = rebuild_with_indices(src_w2e).to(float)
        # del sim_val, sim_id, id_x, ind
        sim = remain_topk_sim(sim, k=sparse_k).to(float)
        if dense_mm:
            tgm = src_w2e.t().to_dense().mm(sim.to_dense())
            tgm = tgm.mm(trg_w2e.to_dense())
        else:
            tgm = spspmm(src_w2e.t(), sim)
            tgm = spspmm(tgm, trg_w2e)
    if do_sinkhorn:
        tgm = sinkhorn_process(tgm)
    return dense_to_sparse(tgm)
