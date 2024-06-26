import requests
from time import sleep

import srsly                           # to decode output
from more_itertools import chunked     # to batch data for API calls



# for type checking
from typing import Optional, Iterable, Dict
from requests.models import Response


class Index:

    def __init__(self, API_key: str, name: str):

        self.API_key = API_key
        self.base_url = "https://sturdystatistics.com/api/text/v1/index"

        self.name = name
        self.id = None

        status = self._get_status(index_name=self.name)
        if status is None:
            self.id = self._create(self.name)
            print(f"""Created new index with id="{self.id}".""")
        else:
            self.id = status["index_id"]
            print(f"""Found an existing index with id="{self.id}".""")


    def _post(self, url: str, params: Dict) -> Response:
        payload = {"api_key": self.API_key, **params}
        res = requests.post(self.base_url + url, json=payload)
        return res

    def _get(self, url: str, params: Dict) -> Response:
        params = {"api_key": self.API_key, **params}
        res = requests.get(self.base_url + url , params=params)
        return res

    def _check_status(self, info: Response) -> None:
        if (200 != info.status_code):
            print(f"""error code {info.status_code}""")
            print(info.content.decode("utf-8"))
        assert(200 == info.status_code)



    def _create(self, index_name: str):
        """Creates a new index. An index is the core data structure for
    storing data.  Once the index is trained, an index may also be
    used to search, query, and analyze data. If an index with the
    provided name already exists, no index will be created and the
    metadata of that index will be returned.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/createIndex

    """

        # Create a new index associated with this API key.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "name": "INDEX_NAME"
        #    }'

        info = self._post("", dict(name=index_name))
        self._check_status(info)

        index_id = info.json()["index_id"]

        return index_id



    def _get_status_by_name(self, index_name: str):

        # List all indices associated with this API key.  Equivalent to:
        #
        # curl -X GET 'https://sturdystatistics.com/api/text/v1/index?api_key=API_KEY'
        #
        # https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/listIndicies

        info = self._get("", dict())
        self._check_status(info)

        # find matches by name
        matches = [ i for i in info.json() if i["name"] == index_name ]
        if (0 == len(matches)):
            return None

        assert(1 == len(matches))
        return matches[0]



    def _get_status_by_id(self, index_id: str):

        # curl -X GET 'https://sturdystatistics.com/api/text/v1/index/{index_id}?api_key=API_KEY'

        info = self._get(f"/{index_id}", dict())
        self._check_status(info)

        status = info.json()
        return status



    def _get_status(self,
                   index_name: Optional[str] = None,
                   index_id: Optional[str] = None):
        """Look up an index by name or ID and return all metadata
    associated with the index.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/getSingleIndexInfo

    """

        if (index_name is None) and (index_id is None):
            raise ValueError("Must provide either an index_name or an index_id.")

        if (index_name is not None) and (index_id is not None):
            raise ValuError("Cannot provide both an index_name and an index_id.")


        if index_id is not None:
            # look up by index_id:
            return self._get_status_by_id(index_id)

        # look up by name:
        return self._get_status_by_name(index_name)



    def get_status(self):
        if self.id is not None:
            return self._get_status(index_id=self.id)
        else:
            return self._get_status(index_name=self.name)



    def commit(self, max_wait_seconds=300):
        """
        """

        print(f"""committing changes to index "{self.id}"...""", end="")

        # Commit changes from the staging index to the permanent index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc/commit \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #    }'

        info = self._post(f"/{self.id}/doc/commit", dict())
        self._check_status(info)


        # poll once per second until status is ready, or until we timeout:
        n_checks = 0
        while n_checks < max_wait_seconds:
            sleep(1)
            status = self.get_status()
            n_checks += 1

            if status["state"] != "committing":
                print("done", end="\n")
                return status


        # we have exceeded the max waiting time
        print("timeout", end="\n")
        return None



    def upload_and_commit(self,
                          records: Iterable[Dict],
                          batch_size: int = 200):
        """Uploads documents to the index and commit them for
    permanent storage.  Documents are processed by the AI model if the
    index has been trained.

    Documents are provided as a list of dictionaries. The content of
    each document must be plain text and is provided under the
    required field doc.  You may provide a unique document identifier
    under the optional field doc_id. If no doc_id is provided, we will
    create an identifier by hashing the contents of the
    document. Documents can be updated via an upsert mechanism that
    matches on doc_id. If doc_id is not provided and two docs have
    identical content, the most recently uploaded document will upsert
    the previously uploaded document.

    This is a locking operation. A client cannot call upload, train or
    commit while an upload is already in progress. Consequently, the
    operation is more efficient with batches of documents. The API
    supports a batch size of up to 250 documents at a time. The larger
    the batch size, the more efficient the upload.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/writeDocs

    """

        status = self.get_status()

        if "untrained" == status["state"]:
            print("Uploading data to UNTRAINED index for training.")
        elif "ready" == status["state"]:
            print("Uploading data to TRAINED index for prediction.")
        else:
            raise RuntimeError(f"""Unknown status "{status['state']}" for index "{self.name}".""")

        results = []

        # Upload docs to the staging index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "docs": JSON_DOC_DATA
        #    }'

        print("uploading data to index...")
        committed = True
        for i, batch in enumerate(chunked(records, batch_size)):
            committed = False
            info = self._post(f"/{self.id}/doc", dict(docs=batch))
            self._check_status(info)
            results.extend(info.json()["results"])
            print(f"""    upload batch {1+i:4d}: response {str(info)}""")

            if 0 == ((i+1) % 10):
                self.commit()
                committed = True

        if not committed:
            self.commit()

        return results



    def train(self, params: Dict, force: Optional[bool] = None):
        """Trains an AI model on all documents in the production
    index. Once an index has been trained, documents are queryable,
    and the model automatically processes subsequently uploaded
    documents.

    The AI model identifies thematic information in documents, permitting
    semantic indexing and semantic search. It also enables quantitative
    analysis of, e.g., topic trends.

    The AI model may optionally be supervised using metadata present in the
    index. Thematic decomposition of the data is not unique; supervision
    guides the model and aligns the identified topics to your intended
    application. Supervision also allows the model to make predictions.

    Data for supervision may be supplied explicitly using the
    label_field_names parameter. Metadata field names listed in this
    parameter must each store data in a ternary true/false/unknown format.
    For convenience, supervision data may also be supplied in a sparse "tag"
    format using the tag_field_names parameter. Metadata field names listed
    in this parameter must contain a list of labels for each document. The
    document is considered "true" for each label listed; it is implicitly
    considered "false" for each label not listed. Consequently, the "tag"
    format does not allow for unknown labels. Any combination of
    label_field_names and tag_field_names may be supplied.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/trainIndex

    """

        status = self.get_status()

        if ("untrained" != status["state"]) and not force:
            print(f"index {self.name} is already trained.")
            return status

        # Issue a training command to the index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/train \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      PARAMS
        #    }'

        info = self._post(f"/{self.id}/train", params)
        self._check_status(info)

        return info.json()



    def predict(self, records: Iterable[Dict], batch_size: int = 200):
        """"Predict" function analogous to sklearn or keras: accepts
    a batch of documents and returns their corresponding predictions.

    Performs an upload operation with `save=false` and without a commit step.
    This function does not mutate the index in any way.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/writeDocs

    """

        status = self.get_status()

        if "ready" != status["state"]:
            raise RuntimeError(f"""Cannot run predictions on index "{self.name}" with state {status["state"]}.""")


        results = []

        # Upload docs to the staging index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "save": "false",
        #      "docs": JSON_DOC_DATA
        #    }'

        print("running predictions...")
        for i, batch in enumerate(chunked(records, batch_size)):
            info = self._post(f"/{self.id}/doc", dict(docs=batch, save="false"))
            self._check_status(info)
            results.extend(info.json()['results'])
            print(f"""    upload batch {1+i:4d}: response {str(info)}""")
            print("...done")

            # no commit needed since this makes no change to the index

        return results
