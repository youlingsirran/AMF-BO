import pandas as pd
from tqdm import tqdm

from amfbo.extraction.drain.drain_template_extractor import *


def processing_feature(svc, log, miner):
    cluster = miner.match(log)
    if cluster is None:
        eventId = -1
    else:
        eventId = cluster.cluster_id
    res = {'service':svc,'id':eventId, 'count':1}
    return res

def extract_log_events(log_df: pd.DataFrame, miner: drain3.TemplateMiner, low_freq_p: float):
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=False)
    err_keywords = ['error', 'fail', 'exception']
    select_events = ['-1']
    for idx, c in enumerate(sorted_clusters):
        if idx < int(low_freq_p * len(sorted_clusters)):
            # low-frequency templates
            select_events.append(c.cluster_id)
            continue
        for keyword in err_keywords:
            if keyword in c.get_template().lower():
                # error log templates
                select_events.append(c.cluster_id)

    log_df.sort_values(by=['timestamp'], ascending=True, inplace=True)
    logs=log_df['message'].values
    svcs=log_df['service'].values

    events_dict = {'service':[], 'id': [], 'count':[]}
    for i,log in tqdm(enumerate(logs)):
        res=processing_feature(svcs[i], log, miner)
        events_dict['service'].append(res['service'])
        events_dict['id'].append(res['id'])
        events_dict['count'].append(res['count'])
    event_df=pd.DataFrame(events_dict)
    event_df = event_df[event_df['id'].isin(select_events)]
    event_gp = event_df.groupby(['id', 'service'])
    events=[[svc, str(event_id)] for (event_id, svc), _ in event_gp]
    return events

if __name__ == '__main__':
    log_df = pd.DataFrame({
        'timestamp': [1, 2, 3, 4, 5],
        'message': [
            'User login success',
            'ERROR: connection failed',
            'User logout',
            'Exception in thread',
            'User login success'
        ],
        'service': ['web', 'db', 'web', 'db', 'web']
    })

    miner = TemplateMiner()
    for msg in log_df['message']:
        miner.add_log_message(msg)
    low_freq_p = 0.2
    events = extract_log_events(log_df, miner, low_freq_p)
    print(events)


    # from drain3 import TemplateMiner
    # from drain3.template_miner_config import TemplateMinerConfig
    # import json
    # config = TemplateMinerConfig()
    #
    # template_miner = TemplateMiner(config=config)
    # log_line = "2023-01-01 12:00:00 ERROR Failed to connect to DB at 192.168.1.1"
    # result = template_miner.add_log_message(log_line)
    #
    # print(f"Template: {result['template_mined']}")
    # print(f"Cluster ID: {result['cluster_id']}")
    # clusters = template_miner.drain.clusters
    # for cluster in clusters:
    #     print(f"Cluster {cluster.cluster_id}: {cluster.get_template()}")
    # state = template_miner.get_state()
    # with open("drain3_state.json", "w") as f:
    #     json.dump(state, f)
    #
    # with open("drain3_state.json", "r") as f:
    #     state = json.load(f)
    # template_miner = TemplateMiner()
    # template_miner.load_state(state)