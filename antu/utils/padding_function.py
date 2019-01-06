

def shadow_padding(batch_input, vocabulary):
    maxlen = 0
    for ins in batch_input:
        for field_name in ins:
            if isinstance(ins[field_name], dict):
                for vocab_name in ins[field_name]:
                    maxlen = max(maxlen, len(ins[field_name][vocab_name]))
            else:
                maxlen = max(maxlen, len(ins[field_name]))

    masks = dict()
    inputs = dict()
    for ins in batch_input:
        for field_name in ins:
            if isinstance(ins[field_name], list):
                if field_name not in masks:
                    masks[field_name] = list()
                if field_name not in inputs:
                    inputs[field_name] = list()
                padding_length = maxlen - len(ins[field_name])
                inputs[field_name].append(ins[field_name] + [0] * padding_length)
                ins_mask = [1]*(maxlen-padding_length) + [0]*padding_length
            else:
                if field_name not in masks:
                    masks[field_name] = dict()
                if field_name not in inputs:
                    inputs[field_name] = dict()
                for vocab_name in ins[field_name]:
                    padding_length = maxlen - len(ins[field_name][vocab_name])
                    if vocab_name not in vocabulary.no_pad_namespace:
                        padding_index = vocabulary.get_padding_index(vocab_name)
                    else: padding_index = 0
                    padding_list = [padding_index] * padding_length
                    ins_input = ins[field_name][vocab_name] + padding_list
                    ins_mask = [1]*(maxlen-padding_length) + [0]*padding_length
                    if vocab_name not in masks[field_name]:
                        masks[field_name][vocab_name] = list()
                    if vocab_name not in inputs[field_name]:
                        inputs[field_name][vocab_name] = list()
                    masks[field_name][vocab_name].append(ins_mask)
                    inputs[field_name][vocab_name].append(ins_input)
    return inputs, masks