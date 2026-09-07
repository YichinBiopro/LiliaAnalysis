"""Complete pre/post intervals on the recording's physical time axis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lilia.windowing import continuous_slices, time_slice


@dataclass(frozen=True)
class EventWindow:
    name: str
    onset_us: int
    pre: slice
    post: slice
    segment_id: int


def select_event_windows(time_us, fs, events, pre_sec, post_sec=None, *, segment_ids=None, segment_end_us=None):
    """Return accepted windows and an audit row for every requested event.

    ``events`` contains (name, absolute onset in microseconds). Both half-open
    intervals must fit in the same continuous segment, including their physical
    boundaries. Never clip incomplete intervals or move an onset out of a gap.
    """
    post_sec = pre_sec if post_sec is None else post_sec
    if not all(np.isfinite(v) and v > 0 for v in (fs, pre_sec, post_sec)):
        raise ValueError('fs and event durations must be finite and positive')
    t = np.asarray(time_us)
    if t.dtype.kind not in 'iu' or not len(t):
        raise ValueError('Event timestamps must be nonempty integer microseconds')
    segments = continuous_slices(t, fs, segment_ids=segment_ids)
    period = int(round(1e6 / fs))
    pre_us, post_us = int(round(pre_sec * 1e6)), int(round(post_sec * 1e6))
    if min(pre_us, post_us, period) < 1:
        raise ValueError('Event durations and sample period must span microseconds')
    ids = [i if segment_ids is None else int(segment_ids[sl.start]) for i, sl in enumerate(segments)]
    ends = [min(int(t[sl.stop - 1]) + period, (segment_end_us or {}).get(sid, int(t[sl.stop - 1]) + period))
            for sid, sl in zip(ids, segments)]
    accepted, audit = [], []
    for name, onset_us in events:
        onset_us = int(onset_us)
        lo, hi = onset_us - pre_us, onset_us + post_us
        row = dict(Event=name, Onset_US=onset_us, Pre_Start_US=lo,
                   Post_End_US=hi, Window_Size=float(pre_sec),
                   Post_Window_Size=float(post_sec), Status='excluded', Reason='')
        segment_id = next((i for i, sl in enumerate(segments)
                           if int(t[sl.start]) <= onset_us < ends[i]), None)
        if segment_id is None:
            row['Reason'] = ('onset_outside_recording' if onset_us < int(t[0]) or
                             onset_us >= ends[-1] else 'onset_in_gap')
        else:
            sl = segments[segment_id]
            if lo < int(t[sl.start]):
                row['Reason'] = 'pre_outside_recording' if segment_id == 0 else 'pre_crosses_gap'
            elif hi > ends[segment_id]:
                row['Reason'] = ('post_outside_recording' if segment_id == len(segments) - 1
                                 else 'post_crosses_gap')
            else:
                pre, post = time_slice(t, lo, onset_us), time_slice(t, onset_us, hi)
                if pre.start == pre.stop or post.start == post.stop:
                    row['Reason'] = 'empty_interval'
                else:
                    accepted.append(EventWindow(name, onset_us, pre, post, ids[segment_id]))
                    row.update(Status='accepted', Segment_ID=ids[segment_id],
                               Pre_Start_Idx=pre.start, Pre_End_Idx=pre.stop,
                               Post_Start_Idx=post.start, Post_End_Idx=post.stop)
        audit.append(row)
    return accepted, audit
