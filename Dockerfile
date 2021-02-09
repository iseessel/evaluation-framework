# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
FROM tensorflow/tensorflow:latest-gpu

ENV GOOGLE_APPLICATION_CREDENTIALS="service-account.json"

# Installs necessary dependencies.
RUN pip install pandas
RUN pip install google-cloud-bigquery
RUN pip install pyarrow
RUN pip install tensorflow-gpu
RUN pip install google-cloud-bigquery-storage
RUN pip install sklearn

COPY . .

CMD ["python3", "boosted_tree_features_v10.py" ]
