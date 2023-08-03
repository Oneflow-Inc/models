# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

domain_list = {
    "mnli": ['fiction', 'government', 'slate', 'telephone', 'travel'],
    "senti": ['books', 'dvd', 'electronics', 'kitchen'],
}

domain_to_id = {
    "mnli": {
        'fiction': 0,
        'government': 1,
        'slate': 2,
        'telephone': 3,
        'travel': 4,
    },
    "senti": {
        'books': 0,
        'dvd': 1,
        'electronics': 2,
        'kitchen': 3,
    }
}

label_map = {
    "mnli": {},
    "senti": {'positive': 0, 'negative': 1},
}