from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import sort_edge_index

from eval import *
from graph_utils import get_mask_with_sim
from text_sim import *
from transformer_helper import BERT, EmbeddingLoader
from tqdm import tqdm
import random
import time

# REMAINS = [
#     "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
#     "http://www.w3.org/2001/XMLSchema#integer",
#     "http://www.w3.org/2001/XMLSchema#double",
# ]

def get_name_feature_map_trans(sents, embedding_loader=None, device='cpu',
                         batch_size=2048, use_fasttext=False, lang=None, trans=None,
                         **kwargs):
    word_id_map = {}
    entity2word = []
    word2entity = collections.defaultdict(set)
    # device = torch.device(kwargs.get('device', 'cpu'))
    # batch_size = kwargs.get('batch_size', 2048)
    tokenizer = None if embedding_loader is None else embedding_loader.tokenizer

    for ent_id, sent in enumerate(sents):
        entity2word.append([])
        for word in [trans[i] if i in trans.keys() else i for i in tokenize(sent, tokenizer)]:
            word_id_map, word_id = add_cnt_for(word_id_map, word)
            entity2word[-1].append(word_id)
            word2entity[word_id].add(ent_id)
    word2entity = [word2entity[i] for i in range(len(word_id_map))]
    words = mp2list(word_id_map)
    if use_fasttext:
        if isinstance(lang, str):
            embeddings = get_fasttext_aligned_vectors(words, device, lang)
        else:
            embeddings = torch.cat([get_fasttext_aligned_vectors(words, device, lang) for lang in lang], dim=1)
    else:
        i = 0
        all_embed = []
        embed_size = 0
        lens = []
        print("sents:",len(sents))
        while i < len(sents):
            embed, length = embedding_loader.get_embed_list(sents[i:min(i + batch_size, len(sents))], True)
            i += batch_size
            embed_size = embed.size(-1)
            lens.append(length.cpu().numpy())
            all_embed.append(embed.cpu().numpy())
        vectors = [emb for batch in all_embed for emb in batch]
        lens = [l for batch in lens for l in batch]
        vectors = [vectors[i][:lens[i]] for i in range(len(vectors))]
        embeddings = torch.zeros([len(words), embed_size], device=device, dtype=torch.float)
        for i, ent in enumerate(entity2word):
            index = torch.tensor(ent, device=device, dtype=torch.long)
            embeddings[index] += torch.tensor(vectors[i]).to(device)[:len(ent)]
            if i % 5000 == 0:
                print("average token embed --", i, "complete")
        if kwargs.get('size_average', True):
            sizes = torch.tensor([len(i) for i in word2entity]).to(device)
            print(sizes)
            embeddings /= sizes.view(-1, 1)
    if kwargs.get('normalize', True):
        embeddings = normalize_vectors(embeddings, kwargs.get('center', True))
    return words, embeddings.to(device), word2entity, entity2word

class DBP15k(InMemoryDataset):
    r"""The DBP15K dataset from the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding"
    <https://arxiv.org/abs/1708.05045>`_ paper, where Chinese, Japanese and
    French versions of DBpedia were linked to its English version.
    Node features are given by pre-trained and aligned monolingual word
    embeddings from the `"Cross-lingual Knowledge Graph Alignment via Graph
    Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        pair (string): The pair of languages (:obj:`"en_zh"`, :obj:`"en_fr"`,
            :obj:`"en_ja"`, :obj:`"zh_en"`, :obj:`"fr_en"`, :obj:`"ja_en"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    file_id = '1dYJtj1_J4nYJdrDY95ucGLCuZXDXI7PL'

    def __init__(self, root, pair, device='cpu',
                 transform=None, pre_transform=None,
                 init_type="MNEAP-L",
                 do_sinkhorn=False):
        assert pair in DBP15K_PAIRS
        self.pair = pair
        self.device = device
        self.init_type = init_type
        self.do_sinkhorn = do_sinkhorn
        super(DBP15k, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return DBP15K_PAIRS

    @property
    def processed_file_names(self):
        return '{0}.pt'.format(self.pair)

    def download(self):
        pass

    @torch.no_grad()
    def process(self):

        g1_path = osp.join(self.raw_dir, self.pair, 'triples_1')
        g2_path = osp.join(self.raw_dir, self.pair, 'triples_2')
        x1_path = osp.join(self.raw_dir, self.pair, 'id_features_1')
        x2_path = osp.join(self.raw_dir, self.pair, 'id_features_2')

        attrgnn_name2id_path = [osp.join(self.raw_dir, self.pair, "entity2id_" + lang + ".txt")
                                for lang in self.pair.split("_")]
        n2i_attrgnn = tuple([self.read_name2id_file(path) for path in attrgnn_name2id_path])

        edge_index1, rel1, assoc1, name_words1 = self.process_graph(
            g1_path, x1_path)
        edge_index2, rel2, assoc2, name_words2 = self.process_graph(
            g2_path, x2_path)

        name_paths = [osp.join(self.raw_dir, self.pair, "ent_ids_" + str(i)) for i in range(1, 3)]

        name2id1 = self.get_name2id(name_paths[0], assoc1)
        name2id2 = self.get_name2id(name_paths[1], assoc2)

        n2i_dataset = (name2id1, name2id2)
        saveobj(n2i_dataset, osp.join(self.raw_dir, self.pair, 'ent_name2id'))
        # translates = self.read_translate(n2i_dataset)
        # name_words1, name_words2 = translates
        total_y = self.process_y_n2id(osp.join(self.raw_dir, self.pair, "entity_seeds.txt"),
                                      n2i_attrgnn, n2i_dataset)
        hard_y, hard_val_y, hard_train_y = tuple(
            [self.process_y_n2id(osp.join(self.raw_dir, self.pair, ty + "_entity_seeds.txt"),
                                 n2i_attrgnn, n2i_dataset) for ty in ["test", "valid", "train"]])
        train_y, test_y, val_y = random_split(total_y, device=self.device)
        ground_truths = [total_y]  # , test_y, hard_y]
        
        t1=time.perf_counter()
        
        if self.init_type=="NEAP":
            embedding_loader = EmbeddingLoader('bert-base-cased')
            g1_words, g1_emb, g1_w2e, g1_e2w = \
                get_name_feature_map(name_words1, embedding_loader, device=self.device, normalize=False)
            g2_words, g2_emb, g2_w2e, g2_e2w = \
                get_name_feature_map(name_words2, embedding_loader, device=self.device, normalize=False)
            tokenizer = embedding_loader.tokenizer
            #print(len(g1_words), len(g2_words))
            g1_tfidf, g2_tfidf = self.get_tf_idf(g1_words, name_words1, tokenizer), \
                                 self.get_tf_idf(g2_words, name_words2, tokenizer)
            del embedding_loader

            rawsim_x2y = token_level_similarity(g1_tfidf, g2_tfidf, g1_emb, g2_emb, sparse_k=1, do_sinkhorn=False)
            rawsim_y2x = token_level_similarity(g2_tfidf, g1_tfidf, g2_emb, g1_emb, sparse_k=1, do_sinkhorn=False)
            
        elif self.init_type=="MNEAP-L" or self.init_type=="MNEAP-H":
            from character import get_character_tf_idf
            import unicodedata
            for i in range(len(name_words1)):
                name_words1[i]=remove_punc(name_words1[i]).lower()
            for i in range(len(name_words2)):
                name_words2[i]=remove_punc(name_words2[i]).lower()
            g1_token = list(set([token for words in name_words1 for token in tokenize(words, None)]))
            g2_token = list(set([token for words in name_words2 for token in tokenize(words, None)]))
            g0_character = list(set([character for token in (g1_token+g2_token) for character in list(enumerate(list(token)))]))
            g1_token_tfidf, g2_token_tfidf = self.get_tf_idf(g1_token, name_words1, None), \
                                 self.get_tf_idf(g2_token, name_words2, None)
            g1_character_tfidf, g2_character_tfidf = get_character_tf_idf(g0_character, g1_token), \
                                 get_character_tf_idf(g0_character, g2_token)
            g1_character_tfidf, g2_character_tfidf = torch.tensor(g1_character_tfidf.A, device=self.device, dtype=torch.float64), \
                                 torch.tensor(g2_character_tfidf.A, device=self.device, dtype=torch.float64)
            g0_character=torch.eye(len(g0_character), len(g0_character), device=self.device, dtype=torch.float64)

            g1_character_tfidf, g2_character_tfidf = g1_character_tfidf.mm(g0_character), \
                                 g2_character_tfidf.mm(g0_character)
            k=1 if self.init_type=="MNEAP-L" else None
            rawsim_x2y = token_level_similarity(g1_token_tfidf, g2_token_tfidf, g1_character_tfidf, g2_character_tfidf, sparse_k=k, do_sinkhorn=False)
            rawsim_y2x = token_level_similarity(g2_token_tfidf, g1_token_tfidf, g2_character_tfidf, g1_character_tfidf, sparse_k=k, do_sinkhorn=False)
            del g0_character,g1_token,g2_token
            g1_emb,g2_emb=g1_character_tfidf,g2_character_tfidf
        
        t2=time.perf_counter()
        
        if self.do_sinkhorn==True:
            sim_x2y, sim_y2x = apply(sinkhorn_process, rawsim_x2y, rawsim_y2x)
        else:
            sim_x2y, sim_y2x = rawsim_x2y, rawsim_y2x
        
        t3=time.perf_counter()
        print("time w/o sinkhorn:",t2-t1)
        print("sim cnt", rawsim_x2y._values().numel(), rawsim_y2x._values().numel())
        print("time:",t3-t1)
        print("sim cnt", sim_x2y._values().numel(), sim_y2x._values().numel())
        #assert(1==2)
        
        x1, x2 = self.bert_encode_maxpool([name_words1, name_words2])
        x1, x2, g1_emb, g2_emb = apply(lambda x: x.to(self.device), x1, x2, g1_emb, g2_emb)
        
        lev_x2y = pairwise_edit_distance(name_words1, name_words2)
        lev_y2x = lev_x2y.t()
        sims = save_similarity_matrix(False, lev_x2y=lev_x2y, lev_y2x=lev_y2x, sim_x2y=sim_x2y, sim_y2x=sim_y2x)

        # analyse
        for y in ground_truths:
            print("NEAP-wo sinkhorn")
            evaluate_sim_matrix(y, rawsim_x2y, rawsim_y2x)
            print("NEAP")
            evaluate_sim_matrix(y, sim_x2y, sim_y2x)
            print("EDIT-DIST")
            evaluate_sim_matrix(y, lev_x2y.to(self.device), lev_y2x.to(self.device))
            print("MAXPOOL")
            evaluate_embeds(x1, x2, y)
        # end
        lens = (x1.size(0), x2.size(0))
        data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, x2=x2,
                    edge_index2=edge_index2, rel2=rel2,
                    test_y=test_y, train_y=train_y, val_y=val_y,
                    total_y=total_y, hard_y=hard_y, hard_val_y=hard_val_y, hard_train_y=hard_train_y,
                    **sims)

        torch.save(self.collate([data]), self.processed_paths[0])

    def bert_encode_maxpool(self, sent_list):
        '''
         use BERT to encode sentences
        '''
        bert = BERT()
        bert.to("cpu")
        return [bert.pooled_encode_batched(lst, save_gpu_memory=True) for lst in sent_list]

    def str2id_map_to_list(self, mp):
        return sorted(list(mp.keys()), key=lambda x: mp[x])

    def get_assoc(self, n2i_curr, n2i_dataset):
        assoc = {}
        mx_id = 0
        for name, curr_id in n2i_curr.items():
            mx_id = max(curr_id, mx_id)
            assoc[curr_id] = n2i_dataset[name]

        x = np.zeros([mx_id + 1], np.long)
        for k, v in assoc.items():
            x[k] = v

        return torch.from_numpy(x)

    def process_y_n2id(self, link_path, n2i_curr, n2i_dataset):
        curr_reverse = self.pair[:2] == "en"
        assoc0, assoc1 = tuple([self.get_assoc(n2i_curr[i], n2i_dataset[i]) for i in range(2)])
        g1, g2 = read_txt_array(link_path, "\t", dtype=torch.long).t()
        if curr_reverse:
            g1, g2 = g2, g1
        g1 = assoc0[g1]
        g2 = assoc1[g2]
        return torch.stack([g1, g2], dim=0).to(self.device)

    def read_name2id_file(self, path, name_place=0, id_place=1, split='\t', skip=1, assoc=None):
        name2id = {}
        with open(path, "r") as f:
            for line in f:
                if skip > 0:
                    skip -= 1
                    continue
                info = line.strip().split(split)
                now = int(info[id_place])
                if assoc is not None:
                    now = assoc[now]
                name2id[info[name_place]] = now

        return name2id

    def get_count(self, words, entity_list):
        raw_count = get_count(words, entity_list)
        return to_torch_sparse(raw_count, device=self.device).to(float).t()

    def get_tf_idf(self, words, entity_list, tokenizer):
        raw_tf_idf = get_tf_idf(words, entity_list, tokenizer)
        return to_torch_sparse(raw_tf_idf, device=self.device).to(float).t()

    def get_name2id(self, name_path, assoc: Tensor):
        name2id = {}
        assoc = assoc.cpu().tolist()
        with open(name_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                name2id[info[1]] = assoc[int(info[0])]

        return name2id

    def process_graph(self, triple_path, feature_path):
        g1 = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g1.t()
        name_dict = {}
        with open(feature_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                info = info if len(info) == 2 else info + ['']
                seq_str = remove_punc(info[1]).strip()
                if seq_str == "":
                    seq_str = '<unk>'
                name_dict[int(info[0])] = seq_str

        idx = torch.tensor(list(name_dict.keys()))
        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))

        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)

        # xs = [None for _ in range(idx.size(0))]
        names = [None for _ in range(idx.size(0))]
        for i in name_dict.keys():
            names[assoc[i]] = name_dict[i]
        # x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)

        return edge_index, rel, assoc, names

    def process_y(self, path, assoc1, assoc2):
        row, col, mask = read_txt_array(path, sep='\t', dtype=torch.long).t()
        mask = mask.to(torch.bool)
        return torch.stack([assoc1[row[mask]], assoc2[col[mask]]], dim=0)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pair)
