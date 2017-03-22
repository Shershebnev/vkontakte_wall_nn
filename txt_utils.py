#!coding=utf-8
import glob
import numpy as np

def convert_from_alphabet(a):
    # hacky as hell :)
    if 1040 <= a <= 1103 or a in [[1025, 1105]]:  # А-Яа-яЁё
        return a - 1040  # converting to 0-32
    elif 32 <= a <= 64:  # punctuation
        return a + 66
    else:
        return 111  # for everythin else - 111 for now - outside of 0 - 110

def convert_to_alphabet(c):
    if c == 111:
        return 0
    elif c <= 65:
        return c + 1040
    elif 98 <= c <= 130:
        return c - 66
    else:
        return 0

def encode_text(text):
    return list(map(lambda x: convert_from_alphabet(ord(x)), text))

def decode_text(text):
    return "".join(list(map(lambda x: chr(convert_to_alphabet(x)), text)))

def read_data_files(directory, validation = True):
    coded_text = []
    bookranges = []
    filelist = glob.glob(directory, recursive = True)
    for filename in filelist:
        f = open(filename)
        start = len(coded_text)
        coded_text.extend(encode_text(f.read()))
        end = len(coded_text)
        bookranges.append({"start" : start, "end": end, "name": filename.split("/")[-1]})
        f.close()

    assert len(bookranges) != 0, "No training data found in {}".format(directory)

    total_len = len(coded_text)
    validation_len = 0
    number_of_books1 = 0
    for book in reversed(bookranges):
        validation_len += book["end"] - book["start"]
        number_of_books1 += 1
        if validation_len > total_len // 10:
            break

    validation_len = 0
    number_of_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"] - book["start"]
        number_of_books2 += 1
        if validation_len > 90000:
            break

    number_of_books3 = len(bookranges) // 5

    number_of_books = min(number_of_books1, number_of_books2, number_of_books3)

    if validation and number_of_books > 0:
        cutoff = bookranges[-number_of_books]["start"]
    else:
        cutoff = len(coded_text)
    validation_text = coded_text[cutoff:]
    coded_text = coded_text[:cutoff]
    return coded_text, validation_text, bookranges

def sample_text(probs, amount, alphabet_size):
    p = np.squeeze(probs)
    p[np.argsort(p)[:-amount]] = 0
    p = p / np.sum(p)
    return np.random.choice(alphabet_size, 1, p=p)[0]

def rnn_minibatch_sequencer(data, batch_size, sequence_size, number_of_epochs):
    raw_data = np.array(data)
    data_len = raw_data.shape[0]
    number_of_batches = (data_len - 1) // (batch_size * sequence_size)
    assert number_of_batches > 0, "Not enough data"
    data_len = number_of_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:data_len], [batch_size, number_of_batches * sequence_size])
    ydata = np.reshape(data[1:data_len + 1], [batch_size, number_of_batches * sequence_size])

    for epoch in range(number_of_epochs):
        for batch in range(number_of_batches):
            x = xdata[:, batch * sequence_size : (batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size : (batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch
