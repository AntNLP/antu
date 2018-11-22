# pyAnt
Universal data IO module in NLP tasks (for AntNLP Group)



### Dataset

- `vocabs: Dict[str, Vocabulary]`
- `datasets: Dict[str, List[Instance]]`

- `build_dataset(name: str, fpath: str, reader: DatasetReader) -> None`
- `add_vocabulary(name: str, vocab: Vocabulary)`



### DataReader

- `read()`
- `input_to_instance()`



### Instance

- `to_str()`



### Field

- `get_index()`
- `set_index()`



### Logger

