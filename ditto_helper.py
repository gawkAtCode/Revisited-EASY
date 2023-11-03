from collections import defaultdict
from text_sim import *
import json

def serialized(filepath,name_words1,name_words2):
    ent2rel=defaultdict(dict)
    with open(filepath,"r", encoding="utf-8") as f:
        for line in f.readlines():
            now = [i.split("/")[-1] for i in line.strip().split('\t')]
            if now[1] in ent2rel[now[0]].keys():
                ent2rel[now[0]][now[1]].append(now[2])
            else:
                ent2rel[now[0]][now[1]]=[now[2]]
    name_words1_serialized,name_words2_serialized=["COL title VAL "+i for i in name_words1],["COL title VAL "+i for i in name_words2]
    for i in range(len(name_words1_serialized)):
        for j in ent2rel[name_words1[i]].keys():
            name_words1_serialized[i]+=" COL "+j+" VAL "+" ".join(ent2rel[name_words1[i]][j])
    for i in range(len(name_words2_serialized)):
        for j in ent2rel[name_words2[i]].keys():
            name_words2_serialized[i]+=" COL "+j[0]+" VAL "+" ".join(ent2rel[name_words2[i]][j])
    serialized2id1,serialized2id2={name_words1_serialized[i]:i for i in range(len(name_words1))},{name_words2_serialized[i]:i for i in range(len(name_words2))}
    return name_words1_serialized,name_words2_serialized,serialized2id1,serialized2id2

def saveJSONL(filepath,train,valid,test,seeds_jsonl):
    with open(filepath+"/train.txt","w",encoding="utf-8") as f:
        f.write("\n".join(train))
    with open(filepath+"/valid.txt","w",encoding="utf-8") as f:
        f.write("\n".join(valid))
    with open(filepath+"/test.txt","w",encoding="utf-8") as f:
        f.write("\n".join(test))
    with open("./models/ditto/input/input_small.jsonl","w",encoding="utf-8") as f:
        for entry in seeds_jsonl:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def readJSONL(serialized2id1,serialized2id2):
    ids1,ids2,match_confidence=[],[],[]
    with open("./models/ditto/output/output_small.jsonl","r") as f:
        for line in f.readlines():
            line=json.loads(line)
            id1,id2,conf=serialized2id1[line["left"]["title"]],serialized2id2[line["right"]["title"]],line["match_confidence"]
            ids1.append(id1)
            ids2.append(id2)
            match_confidence.append(conf)
    return ids1,ids2,match_confidence

def getSEED(sim_x2y,name_words1_serialized,name_words2_serialized):
    ids,vls=sim_x2y.coalesce().indices(),sim_x2y.coalesce().values()
    ids_top1,vls_top1=remain_topk_sim(sim_x2y,k=1).coalesce().indices(),remain_topk_sim(sim_x2y,k=1).coalesce().values()
    ids_top2,vls_top2=remain_topk_sim(sim_x2y,k=2).coalesce().indices(),remain_topk_sim(sim_x2y,k=2).coalesce().values()
    seeds,seeds_jsonl=[],[]
    for i in range(ids.shape[1]):
        seeds_jsonl.append([{"title":name_words1_serialized[ids[0,i].item()]},{"title":name_words2_serialized[ids[1,i].item()]}])
    #remove_id=defaultdict(int)
    appear_id=defaultdict(int)
    vls_mean=torch.mean(vls_top1).item()
    for i in range(ids_top1.shape[1]):
        if vls_top1[i].item()>vls_mean:
            label="1"
            appear_id[ids_top1[0,i].item()]=ids_top1[1,i].item()
        else:
            label="0"
            continue
        seeds.append(name_words1_serialized[ids_top1[0,i].item()]+"\t"+name_words2_serialized[ids_top1[1,i].item()]+"\t"+label)

    # for i in range(ids_top2.shape[1]):
        # if appear_id[ids_top2[0,i].item()]!=ids_top2[1,i].item():
            # label="0"
            # seeds.append(name_words1_serialized[ids_top2[0,i].item()]+"\t"+name_words2_serialized[ids_top2[1,i].item()]+"\t"+label)
    return seeds,seeds_jsonl
