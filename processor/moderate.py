from atproto import AsyncClient, client_utils
import datetime
import os
from constants import THRESHOLD
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

client = AsyncClient()
client.configure_proxy_header("atproto_labeler", "did:plc:newitj5jo3uel7o4mnf3vj2o")


async def auth_client():
    # This bit is broken if you restart the thing. You may need to delete session.txt and restart if you get token errors
    try:
        with open("session.txt") as f:
            session_string = f.read()

        await client.login(session_string=session_string)
    except:
        await client.login(os.environ["BSKY_USERNAME"], os.environ["BSKY_PASSWORD"])

        session_string = client.export_session_string()
        with open("session.txt", "w") as f:
            f.write(session_string)


async def create_label(result):
    try:
        commit = result["commit"]
        uri = f"at://{result['did']}/{commit['collection']}/{commit['rkey']}"
        cid = commit["cid"]

        blob_cids = []
        add_labels = []
        for image in result["image_results"]:
            for label, score in image["labels"].items():
                if float(score) >= THRESHOLD:
                    # if label == "news" and float(score) >= 0.98:
                    if label == "news" or label == 'newsmedia':
                        pass

                        # add_labels.append("newsmedia-screenshot")
                        # blob_cids.append(image["blob_cid"])
                    elif label != "negative":
                        add_labels.append(f"{label}-screenshot")
                        blob_cids.append(image["blob_cid"])

        if len(add_labels) > 0:
            data = {
                "event": {
                    "$type": "tools.ozone.moderation.defs#modEventLabel",
                    "createLabelVals": add_labels,
                    "negateLabelVals": [],
                    "comment": f"model:howdyaendra/swin_s3_base_224-xblockm-timm",
                },
                "subject": {
                    "$type": "com.atproto.repo.strongRef",
                    "uri": uri,
                    "cid": cid,
                },
                "createdBy": client._session.did,
                "createdAt": datetime.datetime.now().isoformat(),
                "subjectBlobCids": blob_cids,
            }

            print(data)
            return await client.tools.ozone.moderation.emit_event(data=data)
    except Exception as e:
        print(e)
