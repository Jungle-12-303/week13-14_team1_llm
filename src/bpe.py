# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


# 앞에 b 붙이면 bytes 객체로 확정
PAD_TOKEN = b"<pad>" # 패딩
UNK_TOKEN = b"<unk>" # 모르는 단어
BOS_TOKEN = b"<bos>" # 문장 시작
EOS_TOKEN = b"<eos>" # 문장 끝

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
        self.id_to_token = {} # 디코딩에서 사용
        self.token_to_id = {} # 인코딩에서 사용
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        self.id_to_token = {}
        self.token_to_id = {}

        # id_to_token, token_to_id 등록
        # 고정 ID 0~3 등록
        for token_id, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # ID 4~259 <-> byte 0~255 등록
        for i in range(NUM_BYTES):
            token = bytes([i])
            token_id = i + BYTE_OFFSET

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
            
        return

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

        self.merges = []
        self._init_special_tokens()

        tokens = self._text_to_byte_tokens(corpus)

        while len(self.id_to_token) < self.vocab_size:

            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break
            
            best_pair = max(pair_counts, key=lambda pair: pair_counts[pair])
            
            left, right = best_pair
            new_token = left + right

            new_token_id = len(self.id_to_token)

            self.merges.append(best_pair)
            

            self.id_to_token[new_token_id] = new_token
            self.token_to_id[new_token] = new_token_id
            
            tokens = self._merge_pair(tokens, best_pair)
        
        


    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """


        # 1. 데이터를 json 형식에 맞게 변형
        # {
        #     "vocab_size": voca_size,
        #     "id_to_token": {
        #         "type": "bytes" or "tuple" or "int"
        #         "value": token list
        #         ...
        #     },
        #     "merges": [
        #         [{
        #           "type": "bytes" or "tuple" or "int"
        #           "value": left
        #          },
        #          {
        #           "type": "bytes" or "tuple" or "int"
        #           "value": right
        #          }]
        #     ],
        # }

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {
                str(token_id): self._serialize_token(token) for token_id, token in self.id_to_token.items()
            },
            "merges": [
                [self._serialize_token(left), self._serialize_token(right)] for left, right in self.merges
            ]
        }

        # 2. 파일 open 후 저장
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        # 1. 파일 open
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. 데이터 복원
        # {
        #     "vocab_size": voca_size,
        #     "id_to_token": {
        #         "type": "bytes" or "tuple" or "int"
        #         "value": token list
        #         ...
        #     },
        #     "merges": [
        #         [{
        #           "type": "bytes" or "tuple" or "int"
        #           "value": left
        #          },
        #          {
        #           "type": "bytes" or "tuple" or "int"
        #           "value": right
        #          }]
        #     ],
        # }
        self.vocab_size = int(data["vocab_size"])

        
        
        id_token_dict = data["id_to_token"]
        self.id_to_token = {}
        self.token_to_id = {}
        for token_id, token_obj in id_token_dict.items():
            token_id = int(token_id)
            token = self._deserialize_token(token_obj)

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id


        self.merges = []
        for left_obj, right_obj in data["merges"]:
            left = self._deserialize_token(left_obj)
            right = self._deserialize_token(right_obj)
            self.merges.append((left, right))



    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        tokens = self._text_to_byte_tokens(text)

        # merge 작업 추가해야 됨
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)

        ids = []
        for byte in tokens:
            ids.append(self.token_to_id[byte])
        
        if add_bos_eos == True:
            ids.insert(0, self.get_bos_id())
            ids.append(self.get_eos_id())


        return ids



    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        if skip_special == True:
            filtered_ids = []
            for token_id in ids:
                if token_id not in SPECIAL_IDS.values():
                    filtered_ids.append(token_id)
            ids = filtered_ids
        tokens = []
        for token_id in ids:
            tokens.append(self.id_to_token[token_id])

        text = b"".join(tokens).decode("utf-8")

        return text

    
    def _text_to_byte_tokens(self, text: str) -> list[bytes]:
        return [bytes([byte]) for byte in text.encode("utf-8")]
    
    def _merge_pair(self, tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        left, right = pair
        merged = left + right

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def _serialize_token(self, token):
        if isinstance(token, bytes):
            return {
                "type": "bytes",
                "value": list(token),
            }
        if isinstance(token, tuple):
            return {
                "type": "tuple",
                "value": list(token),
            }
        if isinstance(token, int):
            return {
                "type": "int",
                "value": token,
            }

    def _deserialize_token(self, obj):
        if obj["type"] == "bytes":
            return bytes(obj["value"])
        if obj["type"] == "tuple":
            return tuple(obj["value"])
        if obj["type"] == "int":
            return int(obj["value"])
