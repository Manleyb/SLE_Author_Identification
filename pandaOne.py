# for EAP
import pandas as pd
import numpy as np
import chardet
import os
import glob
import xml.etree.ElementTree as ET
from lxml import etree
import csv
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics


# location = '/home/jarvis/Desktop/cse223/PersonalTempFolder/SLE/blogs'
# localDirrectory = os.getcwd()
# localDirrectory


# def blogsinXml(location):


#     return glob(os.path.join(location, "*.xml"))


# filenames = blogsinXml(location)

# for filename in filenames:
#     with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
#         file = open(filename, 'r', encoding='utf-8')
#         result = chardet.detect(f.readline())
#         contents = f.read()
#         soup = BeautifulSoup(contents, 'xml')
#         blogData = soup.find_all('post')
#         posts = []
#         date = soup.find_all('date')

#     for n, title in enumerate(blogData):
#         print("date:", date[n])
#         print("Blog contains:", title.text)
#         print()
# # content = file.readlines()
# # Combine the lines in the list into a string
# # content = "".join(content)
# # bs_content = bs(content, "lxml")\
dataset_folder = "blogs"

import os
from glob import glob 
def get_all_xml_files_in_folder(dataset_folder):
    return glob(os.path.join(dataset_folder, "*.xml"))


def load_file(filename):
    with open(filename, 'rb') as inf:  
        return inf.read()  


filenames = get_all_xml_files_in_folder(dataset_folder)
filename = filenames[0]

contents = load_file(filename)
contents[:50]
type(contents)
chardet.detect(contents)
limit = 30
encodings = {}

for filename in filenames:
    encodings[filename] = chardet.detect(load_file(filename))
    if limit and len(encodings) >= limit:
        break

soup = BeautifulSoup(contents, 'xml')
posts = [p.contents[0] for p in soup.find_all('post')]
posts[0][:50]


def extract_posts(filename):
    contents = load_file(filename)
    soup = BeautifulSoup(contents, 'xml')
    posts = [p.contents[0] for p in soup.find_all('post') if len(p.contents)]
    return posts


number_read = 0

for filename in filenames:
    try:
        number_read += 1
        if number_read > limit:
            break
        posts = posts + extract_posts(filename)
    except Exception as e:
        print(f"Error with file: {filename}")
        raise e

remove_urlLink_match = re.compile("urlLink")


def postprocess(document):
    document = remove_urlLink_match.sub("", document)
    # Common post-processing is to remove whitespace from start and end
    document = document.strip()
    return document


# For matching lines containing a post's date
date_line_matching = re.compile("^<date>(.*)</date>$")


class Post:
    author_number: int
    gender: str
    age: int
    industry: str
    star_sign: str
    date: str  # Unparsed date string, may not be English or even well formed
    post: str

    def to_dict(self):
        return {
            key: getattr(self, key)
            for key in ['author_number', 'gender', 'age', 'industry', 'star_sign', 'date', 'post']
        }

    @staticmethod
    def load_from_file(filename) -> List["Post"]:
        """Load a single file from the blogs dataset, returning many Posts"""
        # The last element is the file extension, which we don't care about
        # example file format is: 5144.male.25.indUnk.Scorpio.xml
        age, author_number, gender, industry, star_sign = Post.extract_attributes_from_filename(
            filename)
        with open(filename, 'rb') as inf:
            contents = load_file(filename)
            soup = BeautifulSoup(contents, 'xml')
            posts = [postprocess(p.contents[0])
                     for p in soup.find_all('post') if len(p.contents)]
        return posts

    @staticmethod
    def extract_attributes_from_filename(filename):
        # Get just the filename component
        base_filename = pathlib.Path(filename).name
        author_number, gender, age, industry, star_sign, _ = base_filename.split(
            ".")
        author_number = int(author_number)
        age = int(age)
        return age, author_number, gender, industry, star_sign

    @staticmethod
    def create_from_attributes(author_number, gender, age, industry, star_sign, date, post):
        """Creates a Post from a set of attributes"""
        # Hint: You could use many other methods of creating this object. I went with this one as it is
        # easy to read and follow.
        p = Post()
        p.author_number = author_number
        p.gender = gender
        p.age = age
        p.industry = industry
        p.star_sign = star_sign
        p.date = date  
        p.post = post 
        return p


filename_id_pattern = re.compile(r"(\d{3,})\..*\..*\..*\..*\.xml")


def load_dataset_from_raw(dataset_folder, ids=None):
    """Load blogs authorship dataset"""
    all_posts = []

    for filename in get_all_xml_files_in_folder(dataset_folder):
        # Either ids is None, in which case we load everything, or ids is given, in which case we only load
        # files containing that id.
        if ids is None or get_filename_id(filename) in ids:
            current_posts = Post.load_from_file(filename)
            all_posts.extend(current_posts)
    return all_posts


def get_all_xml_files_in_folder(dataset_folder):
    return glob(os.path.join(dataset_folder, "*.xml"))


def get_filename_id(filename):
    """Extracts just the id number from filenames of the type "5114.male.25.indUnk.Scorpio.xml"
    """
    # We use search not match, as we don't care if its not the whole string
    match = filename_id_pattern.search(filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not find an ID in filename {filename}")


def save_dataset(all_posts, output_file):
    dataset = pd.DataFrame([post.to_dict() for post in all_posts])
    dataset.to_parquet(output_file, compression='gzip')
    return dataset


def load_dataset(input_file):
    return pd.read_parquet(input_file)


dataset_filename = "blogs_processed.parquet"

all_posts_raw = load_dataset_from_raw(dataset_folder)

save_dataset(all_posts_raw, dataset_filename)
all_posts = load_dataset(dataset_filename)


def get_sampled_authors(dataset, sample_authors):
    mask = dataset['author_number'].isin(sample_authors)
    return dataset[mask]


sample = get_sampled_authors(all_posts, [3574878, 2845196, 3444474, 3445677, 828046,
                                         4284264, 3498812, 4137740, 3662461, 3363271])
sample.sample(10)

documents_train, documents_test, authors_train, authors_test = train_test_split(
    documents, authors)


model = SGDClassifier()
model.fit(X_train, authors_train)


authors_predicted = model.predict(X_test)

print(classification_report(authors_test, authors_predicted))
