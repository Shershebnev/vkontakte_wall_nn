#!coding=utf-8
"""This module downloads all the posts from specified vk group (via group id)
and filters out those, that have non-Russian letters, saving the rest to the
file.
"""
import itertools
import os
import re
import warnings

import vk  # https://github.com/dimka665/vk

MAX_READING_SIZE = 100

def downloader(group_id, output_dir, batch_size):
    """Saves all posts from group_id vk group into the file and returns
    posts-based alphabet

    Args:
        group_id: vk group id. Can be found by looking at the Members link - 
                  the last digits in the link is group id
        output_dir: path to directory to save batches of output files
        batch_size: size of the batch file, i.e. number of posts in one batch.

    Returns:
        set of all symbols that are present in the posts
    """
    if not os.path.isdir(output_dir):
        warnings.warn("Output directory doesn't exist, creating it", Warning,
            stacklevel=2)
        os.makedirs(output_dir)
    # all russian letters, including Ё и ё
    alphabet = set([chr(i) for i in itertools.chain(range(1040, 1104),
                                                    [1025, 1105])])
    pattern = re.compile("[a-zA-Z]")
    session = vk.Session()
    api = vk.API(session)
    # can't find a better way right now
    number_of_posts = api.wall.get(owner_id=-group_id, count=1, offset=1)[0]
    all_data_path = os.path.join(output_dir, "all_data.txt")
    all_data_file = open(all_data_path, "w")
    post_count = 0
    # reading and saving all filtered data
    for off in range(1, number_of_posts + 1, MAX_READING_SIZE):
        print("offset is {}".format(off))
        data = api.wall.get(owner_id=-group_id, count=MAX_READING_SIZE, offset=off)
        data = data[1:]  # data[0] is total number of posts in the community == 4175
        for item in data:
            # if the post had only picture attached, for example, the text
            # field will be empty. Don't need those
            if item["text"]:
                # removing <br> tag, which appeared in some posts
                text = item["text"].replace("<br>", " ")
                # and filtering out posts that contain non-cyrillic characters.
                # Probably should extend the pattern for some special letters.
                # Will see how a-zA-Z works
                if not pattern.findall(text):  # empty list, meaning no matches
                    all_data_file.write("{}\n".format(text))
                    # Updating alphabet with punctutation signs, that are
                    # present in posts. I now this is probably not the best
                    # way, but will stick to this for now
                    alphabet.update(set(text))
                    post_count += 1
    all_data_file.close()
    # reshaping it into files with batches of size batch_size
    all_data_file = open(all_data_path)
    batch_number = 1
    line = all_data_file.readline()
    for start in range(0, post_count, batch_size):
        output_file = open(os.path.join(output_dir, 
                           "batch_{}.txt".format(batch_number)), "w")
        for _ in range(start, start + batch_size):
            if not line:
                break  # reached the end of the file
            output_file.write(line)
            line = all_data_file.readline()
        batch_number += 1
    os.remove(all_data_path)
    return alphabet

# alph = downloader(74479926, "data", 100)
