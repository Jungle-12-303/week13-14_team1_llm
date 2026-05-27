# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = {}

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for i, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[i] = token
        for token, idx in SPECIAL_IDS.items():
            self.token_to_id[token] = idx

        for byte_value in range(256):
            token_id = byte_value + 4
            token = bytes([byte_value])

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
        
        #raise NotImplementedError("_init_special_tokens를 구현하세요.")

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        
        ids = [byte + BYTE_OFFSET for byte in corpus.encode("utf-8")]

        if not self.id_to_token:
            self._init_special_tokens()

        while (len(self.id_to_token) < self.vocab_size):
            token_pair = {}

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair not in token_pair:
                    token_pair[pair] = 1
                else:
                    token_pair[pair] += 1
            if not token_pair:
                break

            best_pair = max(token_pair, key=lambda pair: token_pair[pair])
            new_id = len(self.id_to_token)

            new_token = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]

            self.merges[best_pair] = new_id
            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id

            new_ids = []
            i = 0

            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

        #raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        json_export = {
            "merges" : [],
            "id_to_token" : {}
        }

        for (t1, t2), token_id in self.merges.items():
            json_export["merges"].append({
                "pair" : [t1, t2],
                "id" : str(token_id)
            })

        for token_id, token in self.id_to_token.items():
            str_id = str(token_id)

            if isinstance(token, str):
                json_export["id_to_token"][str_id] = {
                    "type" : "str",
                    "value" : token
                }
            else:
                json_export["id_to_token"][str_id] = {
                    "type" : "bytes",
                    "value" : token.hex()
                }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_export,f, indent=4)

        #raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = {}

        for item in data["merges"]:
            pair = (int(item["pair"][0]), int(item["pair"][1]))
            self.merges[pair] = int(item["id"])
        for str_id, meta in data["id_to_token"].items():
            token_id = int(str_id)

            if meta["type"] == "str":
                token = meta["value"]
            else:
                token = bytes.fromhex(meta["value"])
            
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        #raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
        
        while True:
            valid_pairs = []
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair in self.merges:
                    valid_pairs.append(pair)
            if not valid_pairs:
                break
            
            best_pair = min(valid_pairs, key=lambda p: self.merges[p])
            new_id = self.merges[best_pair]
            new_ids = []
            i = 0

            while i < len(ids):
                if i < len(ids) -1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            ids = new_ids

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids

        #raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_chunks = []

        for idx in ids:
            token = self.id_to_token[idx]

            if isinstance(token, str):
                if skip_special:
                    continue
                else:
                    byte_chunks.append(token.encode("utf-8"))

            else:
                byte_chunks.append(token)

        sum_bytes = b"".join(byte_chunks)

        return sum_bytes.decode("utf-8", errors='replace')

        #raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
