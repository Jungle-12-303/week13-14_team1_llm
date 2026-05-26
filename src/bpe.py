# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


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
        self.merges = []

    def _init_special_tokens(self):
        """
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        # 특수 토큰을 일단 dict에 넣는다
        for i in range(len(SPECIAL_TOKENS)):
            target_token = SPECIAL_TOKENS[i]
            self.id_to_token[i] = target_token
            self.token_to_id[target_token] = i

        # 나머지 259까지는 일반 토큰이다
        # 255개의 낱글자이다: 이후 조합으로 단어 생성됨
        for i in range(len(SPECIAL_TOKENS), 260):
            byte_cnt = i - 4
            self.id_to_token[i] = bytes([byte_cnt])
            self.token_to_id[bytes([byte_cnt])] = i


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

    def train(self, corpus: str):   # corpus: 통째로 온 긴 원본 문장
        """
        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # 초기화 진행
        if not self.id_to_token:
            self._init_special_tokens()

        # 초기 셋팅
        encoded_corpus = corpus.encode("utf-8")
        sequence = [b + 4 for b in encoded_corpus]
        
        # 1. 반복문으로 단어 쌍(token pair)를 생성한다
        while(len(self.id_to_token) < self.vocab_size):
            counts = {}
            # 2. 매 루프 마다 최대 반복 pair 갱신
            for i in range(1, len(sequence)):
                pair = (sequence[i - 1], sequence[i])
                counts[pair] = counts.get(pair, 0) + 1

            # 탈출조건: sequence를 다 소진했을 경우
            if not counts:
                break

            # 3. 최대 반복 pair 찾기 & 바이트로 변환
            max_pair = max(counts, key=counts.get)
            token1 = self.id_to_token[max_pair[0]]
            token2 = self.id_to_token[max_pair[1]]

            # 4. 토큰 자료구조 갱신
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = token1 + token2
            self.token_to_id[token1 + token2] = new_id
            self.merges.append((max_pair, new_id))

            # 5. sequence 갱신
            new_sequence = []
            n = 0
            while n < len(sequence) - 1:
                if max_pair[0] == sequence[n] and max_pair[1] == sequence[n + 1]: 
                    new_sequence.append(new_id)
                    n += 2
                else:
                    new_sequence.append(sequence[n])
                    n += 1

            # 6. 예외 상황 보완
            if n == len(sequence) - 1:
                new_sequence.append(sequence[n])

            sequence = new_sequence
                

        

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
