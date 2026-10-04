"""Exact local max/OR updates; IDs group writes, never selection or features."""
import numpy as np

from .retrieval_paged_names_v170 import DiskVector

MAX_UPDATE_CHUNK = 131072


class GroupedProofVector(DiskVector):
    def maximum(self, ids, values):
        if (self.dtype != np.float64 or values.dtype != np.float64 or values.shape != ids.shape
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
            raise ValueError('Exact finite float64 proof maximum required')
        self.check_ids(ids)  # Every reference, before any grouping or skip.
        # Signed zero tie bits are order-sensitive. They never arise from
        # positive coverage counts, but retain old exact API behavior too.
        if np.any((values == 0) & np.signbit(values)):
            return super().maximum(ids, values)
        with self.mapped(True) as array:
            for start in range(0, len(ids), MAX_UPDATE_CHUNK):
                end = min(start+MAX_UPDATE_CHUNK, len(ids))
                order = np.argsort(ids[start:end], kind='stable')
                positions = ids[start:end][order]
                scores = values[start:end][order]
                begins = np.r_[0, np.flatnonzero(positions[1:] != positions[:-1])+1]
                unique = positions[begins]
                maxima = np.maximum.reduceat(scores, begins)
                array[unique] = np.maximum(array[unique], maxima)

    def flags(self, ids, values):
        if self.dtype != np.bool_ or values.dtype != np.bool_ or values.shape != ids.shape:
            raise ValueError('Exact original proof flags required')
        self.check_ids(ids)  # FALSE is not permission to ignore an invalid reference.
        with self.mapped(True) as array:
            for start in range(0, len(ids), MAX_UPDATE_CHUNK):
                end = min(start+MAX_UPDATE_CHUNK, len(ids))
                positive = values[start:end]
                if np.any(positive):
                    positions = np.unique(ids[start:end][positive])
                    array[positions] = True
                # OR FALSE is identity; avoid touching/writing its pages.
                # All refs and the complete owned file/header still validated.
