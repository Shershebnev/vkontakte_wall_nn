#!coding=utf-8
"""This module downloads all the posts and/or comments from specified vk group
(via group id) and filters out those, that have non-Russian letters, saving the
rest to the file.
"""
import itertools
import os
import re
import time
import warnings

import vk  # https://github.com/dimka665/vk

nrp = "[^а-яА-ЯёЁ{}]".format("".join([chr(i) for i in range(32, 65)]))
nonrussian_pattern = re.compile(nrp)

# english letters pattern
pattern = re.compile("[a-zA-Z]")
MAX_READING_SIZE = 100

def parse_comments(api_conn, data_file_conn, group_id, post_id, comment_count):
    count = 0
    for comm_off in range(1, comment_count + 1, MAX_READING_SIZE):
        comments = api_conn.wall.getComments(owner_id=-group_id, post_id=post_id,
            count=MAX_READING_SIZE, offset=comm_off)
        comments = comments[1:]
        for comment in comments:
            if comment["text"]:
                text = comment["text"].replace("<br>", " ")
                if not nonrussian_pattern.findall(text):
                    data_file_conn.write("{}\n".format(text.strip()))
                    count += 1
        time.sleep(0.3)  # added to avoid request limits
    return count


def downloader(group_id, output_dir, batch_size, posts = True, comments = False):
    """Saves all posts from group_id vk group into the file and returns
    posts-based alphabet

    Args:
        group_id: vk group id. Can be found by looking at the Members link - 
                  the last digits in the link is group id
        output_dir: path to directory to save batches of output files
        batch_size: size of the batch file, i.e. number of posts in one batch.
        posts: whether to save posts. DEFAULT: True
        comments: whether to save comments to posts. DEFAULT: False

    Returns:
        set of all symbols that are present in the posts
    """
    if not os.path.isdir(output_dir):
        warnings.warn("Output directory doesn't exist, creating it", Warning,
            stacklevel=2)
        os.makedirs(output_dir)
    # all russian letters, including Ё и ё, and punctuation
    alphabet = set([chr(i) for i in itertools.chain(range(1040, 1104),
                                                    [1025, 1105],
                                                    range(32, 65))])
    session = vk.Session()
    api = vk.API(session)
    # can't find a better way right now
    number_of_posts = api.wall.get(owner_id=-group_id, count=1, offset=1)[0]
    all_data_path = os.path.join(output_dir, "all_data.txt")
    all_data_file = open(all_data_path, "w")
    data_count = 0
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
                if posts and not nonrussian_pattern.findall(text):
                    all_data_file.write("{}\n".format(text.strip()))
                    data_count += 1
                comment_count = item["comments"]["count"]
                if comments and comment_count > 0:
                    comm_count = parse_comments(api, all_data_file, group_id,
                        item["id"], comment_count)
                    data_count += comm_count
        time.sleep(0.3)  # added to avoid request limits
    all_data_file.close()
    # reshaping it into files with batches of size batch_size
    all_data_file = open(all_data_path)
    batch_number = 1
    line = all_data_file.readline()
    for start in range(0, data_count, batch_size):
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

def get_alphabet(filepath):
    f = open(filepath)
    alphabet = set(map(lambda x: x.rstrip(), f.readlines()))
    return alphabet

# alph = downloader(74479926, "data", 100, True, True)
# f = open("data/alphabet.txt", "w")
# f.write("".join(sorted(alph)))
# f.close()
# print(alph)
