# Reddit Source Data

[Reddit](https://www.reddit.com/) is a social news aggregation and discussion website where users can post links and take part in discussions. Until the spring of 2023, Reddit made its data publicly available through an API that many 3rd party developers built upon. The Pushshift effort was a social media data collection, analysis, and archiving platform that collected Reddit data through the API and made it available to researchers (details of this collection can be found in [The Pushshift Reddit Dataset](https://arxiv.org/abs/2001.08435)). Pushshift released hundreds of collected submission and comment and dumps spanning Reddit’s creation in 2006 to the end of the public API in 2023. While these dumps are no longer available from Pushshift, they can still be found in handful of web archives as of the fall of 2023.

Reddit content comes in two flavors related to the nature of the platform: **submissions** and **comments**. Submissions are variably links to articles or other external content, images or videos, or “selftext” (posts with only text written by the submitter to initiate a discussion thread). Comments are user-written dialogue that form a nested, hierarchical, conversational thread discussing a submission. The indeterminate nature of the data allows for a fair amount of freedom when constructing a pretraining dataset, and several variations of the dataset were explored for pretraining data.

# Dataset versions 

At a high level, three architectures of the Reddit dataset were explored (with minor variations within these versions):

* **comment threads**
    * This format of the dataset assembles comments into conversational threads while leaving submissions unconnected. Comments from the same thread are combined into multi-round dialogue composed of multiple user’s utterances. This is done up to a maximum parent depth, and the assembled dialogues only contain a portion of their parent thread (i.e. documents are snippets of complete thread).

* **atomic content**
    * The simplest form, this version of the dataset treats comments and submissions on equal footing and does not assemble comments into conversational threads. Comments and submissions are incorporated as complete documents.

* **complete threads**
    * The most complex and structured form, this version combines submissions and complete comment threads into a single document with code-like indentation indicating the position of a comment in the thread's hierarchy. This version most closely replicates the experience of browsing a reddit thread and places content in its most complete context. All comments (and the originating submission text) from the same thread are incorporated in a single document. 


## comment_threads_v1

This version assembles all possible conversational threads to a depth of 12. This introduces a great deal of duplication in the dataset, since a single comment is potentially included multiple times in separate conversational paths. V1 was used to ablate various filtering and deduping strategies and it exists in several subversions.

## comment_threads_v2

Similar to v1 in most respects, this version dedupes comments at creation time by keeping only the longest conversational thread for each top-level comment. 


## atomic_content_v3

A first experiment with atomic comments.

## complete_threads_codelike_v4

Only one variation of this version was tried. 

## atomic_content_v5   (*included in Dolma v 1.5*)

A refined version of atomic_content_v3, v5 uses different length and selection criteria for comments and submissions.


# Running dataflow scripts

After placing the collected comment and submission dumps in a google cloud bucket, most versions of the dataset build comment and submission data seperately by running build_comment_data.py and build_submission_data.py:


```
python build_comment_data.py \
  --input_gcs_dir ${DATADIR} \ 
  --output_dir ${OUTPUTDIR} \
  --runner DataflowRunner \
  --temp_location ${OUTPUTDIR}/temp \
  --staging_location ${OUTPUTDIR}/staging \
  --project ${PROJECT} \
  --setup_file ./setup.py
```

```
python build_submission_data.py \
  --input_gcs_dir ${DATADIR} \ 
  --output_dir ${OUTPUTDIR} \
  --runner DataflowRunner \
  --temp_location ${OUTPUTDIR}/temp \
  --staging_location ${OUTPUTDIR}/staging \
  --project ${PROJECT} \
  --setup_file ./setup.py
```

The exception is complete_threads_codelike_v4, which is created with a single script:

```
python build_combined_thread_data.py \
  --input_gcs_dir_comments ${DATADIRCOMMENTS} \ 
  --input_gcs_dir_submissions ${DATADIRSUBMISSIONS} \ 
  --output_dir ${OUTPUTDIR} \
  --runner DataflowRunner \
  --temp_location ${OUTPUTDIR}/temp \
  --staging_location ${OUTPUTDIR}/staging \
  --project ${PROJECT} \
  --setup_file ./setup.py
```

Once a dataflow job is running, you can continue to monitor it in the launching terminal or on the [Dataflow service console](https://console.cloud.google.com/dataflow).
