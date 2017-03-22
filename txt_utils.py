#!coding=utf-8
"""Useful functions for converting text files into batches
"""
import glob
import numpy as np
import warnings

def convert_from_alphabet(a):
    """Return numerical representation of the symbol in the range
    [0...ALPHABET_SIZE]

    Args:
        a: symbol

    Returns:
        numeric representation
    """
    # hacky as hell :)
    if 1040 <= a <= 1103:  # А-Яа-я
        return a - 1040  # converting to 0-32
    elif a == 1025:  # Ё
        return 64
    elif a == 1105:
        return 65
    elif 32 <= a <= 64:  # punctuation
        return a + 34
    else:
        return 99

def convert_to_alphabet(c):
    """Converts numeric value into actual unicode value of the symbol

    Args:
        c: numeric representation of the symbol in range [0...ALPHABET_SIZE]

    Returns:
        Unicode-based numeric representation
    """
    if c <= 63:  # А-Яа-я
        return c + 1040
    elif c == 64:  # Ё
        return 1025
    elif c == 65:  # ё
        return 1105
    elif 66 <= c <= 98:  # punctuation
        return c - 34
    elif c == 99:
        return 0

def encode_text(text):
    """Converts characters in the string into their respective numerical
    representations

    Args:
        text: input string

    Returns:
        list of numeric representations of length len(text)
    """
    return list(map(lambda x: convert_from_alphabet(ord(x)), text))

def decode_text(text):
    """Converts list of numeric representations into normal text

    Args:
        text: list of numeric representations

    Returns:
        converted string of length len(text)
    """
    return "".join(list(map(lambda x: chr(convert_to_alphabet(x)), text)))

def read_data_files(directory, validation = True):
    """Convert text files in the directory and optionally split into training
    and validation sets (default behaviour)

    Args:
        directory: glob path to the input files. Example: data/*.txt
        validation: whether to create a validation set

    Returns:
        list with coded_text (training set), validation set and ranges of each
        text
    """
    # strings, converted with encode_text
    coded_text = []
    # coordinates of start and end of corresponding string from coded_text
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

def sample_text(probs, alphabet_size, amount = -1):
    """Creates a random integer in the range [0...alphabet_size] using
    provided probabilities

    Args:
        probs: list of probabilities
        alphabet_size: alphabet size
        amount: number of top probabilities to consider. DEFAULT: -1, i.e. all

    Returns:
        random integer
    """
    # TODO: might try the one with the highest probability, if there is one
    amount = alphabet_size if amount == -1 else amount
    assert 0 < amount <= alphabet_size, "amount should be > 0 and <= alphabet_size"
    p = np.squeeze(probs)
    assert len(p) == alphabet_size, "number of probabilities should be equal \
    to alphabet_size, got {} and {}".format(len(probs), alphabet_size)
    p[np.argsort(p)[:-amount]] = 0
    p = p / np.sum(p)
    return np.random.choice(alphabet_size, 1, p=p)[0]

def rnn_minibatch_sequencer(data, batch_size, sequence_size, number_of_epochs):
    """Divides the data into batches, so that each sequence is continued in the
    next batch.

    Args:
        data: training set
        batch_size: size of the batch
        sequence_size: size of sequences in the batch
        number_of_epochs: number of epochs

    Yields:
        batch of training sequence, batch of target sequence (training
            sequence shifted by 1) and epoch number
    """
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
