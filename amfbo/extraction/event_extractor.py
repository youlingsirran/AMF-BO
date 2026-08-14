import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from amfbo.extraction.log_event_extractor import extract_log_events
from amfbo.extraction.metric_event_extractor import extract_metric_events
from amfbo.extraction.trace_event_extractor import extract_trace_events
from amfbo.extraction.utils import io_utils

data: dict = io_utils.load('D:/data/GAIA-DataSet-main/MicroSS/post-data-10.pkl')
label_df = pd.read_csv('D:/data/GAIA-DataSet-main/MicroSS/gaia.csv', index_col=0)

metric_detectors = io_utils.load('D:/data/GAIA-DataSet-main/MicroSS/detector/metric-detector-strict-host.pkl')
trace_detectors = io_utils.load('D:/data/GAIA-DataSet-main/MicroSS/detector/trace-detector.pkl')

metric_events_dic = defaultdict(list)
trace_events_dic = defaultdict(list)
log_events_dic = defaultdict(list)
metric_costs, trace_costs, log_costs = [], [], []

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = data[idx]
    # extract metric events
    st = time.time()
    metric_events = []
    for pod_host, kpi_dic in chunk['metric'].items():
        kpi_events = extract_metric_events(pod_host, kpi_dic, metric_detectors[pod_host])
        metric_events.extend(kpi_events)
    metric_costs.append(time.time()-st)
    metric_events_dic[idx]=metric_events

    # extract trace events
    st = time.time()
    trace_events = extract_trace_events(chunk['trace'], trace_detectors)
    trace_events_dic[idx] = trace_events
    trace_costs.append(time.time()-st)

    # # extract log events
    # st = time.time()
    print(idx)

metric_time = np.mean(metric_costs)
trace_time = np.mean(trace_costs)
# log_time = np.mean(log_costs)
print(f'the time cost of extract metric events is {metric_time}')
print(f'the time cost of extract trace events is {trace_time}')


io_utils.save_json('D:/data/GAIA-DataSet-main/MicroSS/events/metric/metric.json', metric_events_dic)
io_utils.save_json('D:/data/GAIA-DataSet-main/MicroSS/events/trace/trace.json', trace_events_dic)
