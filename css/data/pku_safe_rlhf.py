from dataclasses import dataclass
from typing import Optional, List

from datasets import load_dataset


@dataclass
class PkuPair:
    prompt: str
    resp0: str
    resp1: str
    safe0: int
    safe1: int
    safer: Optional[int]


class PkuSafeRLHF:
    def __init__(
        self,
        split: str = "train",
        path: str = "PKU-Alignment/PKU-SafeRLHF",
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        col_prompt: str = "prompt",
        col_resp0: str = "response_0",
        col_resp1: str = "response_1",
        col_safe0: str = "is_response_0_safe",
        col_safe1: str = "is_response_1_safe",
        col_safer: str = "safer_response_id",
    ) -> None:
        ds = load_dataset(path, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self._data: List[PkuPair] = []
        for ex in ds:
            prompt = ex[col_prompt]
            r0 = ex[col_resp0]
            r1 = ex[col_resp1]
            safe0 = int(ex[col_safe0])
            safe1 = int(ex[col_safe1])
            safer_raw = ex.get(col_safer, None)
            if safer_raw is None:
                safer = None
            else:
                try:
                    safer = int(safer_raw)
                except Exception:
                    safer = None
            self._data.append(
                PkuPair(
                    prompt=prompt,
                    resp0=r0,
                    resp1=r1,
                    safe0=safe0,
                    safe1=safe1,
                    safer=safer,
                )
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> PkuPair:
        return self._data[idx]

    def iter_pairs(self):
        for x in self._data:
            yield x
