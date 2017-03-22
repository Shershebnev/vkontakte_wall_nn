#!coding=utf-8
"""This module downloads all the posts from specified vk group (via group id)
and filters out those, that have non-Russian letters, saving the rest to the
file.
"""
import itertools
import re

import vk  # https://github.com/dimka665/vk


def downloader(group_id, output_file):
    """Saves all posts from group_id vk group into the file and returns
    posts-based alphabet

    Args:
        group_id: vk group id. Can be found by looking at the Members link - 
                  the last digits in the link is group id
        output_file: path to output file for saving posts

    Returns:
        set of all symbols that are present in the posts
    """
    # all russian letters, including Ё и ё
    alphabet = set([chr(i) for i in itertools.chain(range(1040, 1104),
                                                    [1025, 1105])])
    pattern = re.compile("[a-zA-Z]")
    session = vk.Session()
    api = vk.API(session)
    # can't find a better way right now
    number_of_posts = api.wall.get(owner_id = -group_id, count = 1, offset = 1)[0]
    f = open(output_file, "w")
    for off in range(1, number_of_posts + 1, 100):
        print("offset is {}".format(off))
        data = api.wall.get(owner_id = -group_id, count = 100, offset = off)
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
                    f.write("{}\n".format(text))
                    # Updating alphabet with punctutation signs, that are
                    # present in posts. I now this is probably not the best
                    # way, but will stick to this for now
                    alphabet.update(set(text))
    f.close()
    return alphabet

# alph = downloader(74479926, "./podslushano_nauka_posts.txt")
