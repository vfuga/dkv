import asyncio
import aiohttp  # noqa
from aiohttp import web
from aiohttp.web_request import BaseRequest
import datetime  # noqa
import argparse
import numpy # noqa
import yaml  # noqa 
import uuid  # noqa
from aiologger.loggers.json import JsonLogger, Logger  # noqa
import json
import logging


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s")
parser = argparse.ArgumentParser(
                                prog="Raft",
                                description="Raft Wrapper",
                                epilog=''
                                )
parser.add_argument("-p", "--port", type=int, default=8880)
args = parser.parse_args()
# logger = JsonLogger.with_default_handlers(level=logging.DEBUG)
# logger = Logger.with_default_handlers(name=__name__, level=logging.DEBUG)
logger = logging.getLogger("raft/whoami.py")


with open("./raft.yml") as f:
    raft_config = yaml.safe_load(f)
    raft_config["url-map"] = {}
    raft_config["whoami"] = str(uuid.uuid4())
    raft_config["status"] = "FOLLOWER"
    raft_config["cluster"]["leader"] = None
    logger.info((
        "Config:\n" + json.dumps(raft_config, indent=2, ensure_ascii=False) + "\n"
        ).replace("\n", "\n | "))


async def get_cluster_status():
    """
        - читаем, валидируем конфигурацию кластера
        - проверяем существование узлов, и пытаемся уведомить о  своем присутствии
        - узнаем кто сейчас мастер
    """
    nodes = raft_config["cluster"]["nodes"]

    async def pool(nodes: list[str] = nodes):
        """
            процедура опроса списка узлов реплики
        """
        for url in nodes:
            async with aiohttp.ClientSession() as sess:
                try:
                    yield (url, await asyncio.wait_for(sess.get(url), timeout=5.0))
                except Exception as ex:  # noqa
                    yield (url, None)

    async def notify(nodes: list[str] = nodes):
        """
            процедура оповещения узлов реплики
        """
        payload = {
            "url": raft_config["my-url"],
            "uuid": raft_config["whoami"],
            "status": raft_config["status"],
            "leader": raft_config["cluster"]["leader"]
        }
        for url in nodes:
            if url == raft_config["my-url"]:  # себя уведомлять не надо
                continue
            async with aiohttp.ClientSession() as sess:
                try:
                    await asyncio.wait_for(
                            sess.post(url.strip("/") + "/notify",
                                      json=payload,
                                      headers={"content-type": "application/json"}),
                            timeout=5.0
                    )
                except Exception as ex:  # noqa
                    logger.error(f"{ex}")

    async for r in pool():
        url, cluster_info_response = r
        if cluster_info_response is None:
            logger.warning(f"cannot reach node: {url}")
        else:
            cluster_info_response = await cluster_info_response.json()
            # проверяем конфиг кластера, и пытаемся достучаться до всех узлов
            if tuple(sorted(cluster_info_response["cluster"]["nodes"])) != \
                    tuple(sorted(raft_config["cluster"]["nodes"])):
                raise Exception("Node list was changed!")
            if cluster_info_response["cluster"]["name"] != raft_config["cluster"]["name"]:
                raise Exception("Wrong cluster name!")
            if cluster_info_response["cluster"]["replicaset"] != raft_config["cluster"]["replicaset"]:
                raise Exception("Wrong replicaset!")
            if raft_config["whoami"] == cluster_info_response["whoami"]:
                # если узнал себя в списке хостов
                raft_config["my-url"] = url
                s = json.dumps(cluster_info_response, indent=2)
                logger.info(f"\n# I'm a node [{url}] in the cluster:\n{s}\n".replace("\n", "\n | "))
            else:
                # если это другой узел
                node_st = cluster_info_response["status"]
                node_uuid = cluster_info_response["whoami"]
                node_leader = cluster_info_response["cluster"]["leader"]
                logger.info(f"discovered node: {node_st}, {node_uuid}, {url}")
                raft_config["url-map"][url] = {
                    "uuid": node_uuid,
                    "status": node_st,
                    "leader": node_leader
                }
            continue
    # нотификация - к этому моменту мы знаем все uuid + url узлов в репликасете
    # знаем себя в кластере и других
    # нужно/можно уведомить всех о своем присутствии
    await notify()


async def handler(request: BaseRequest):
    logger.info(request.url)
    logger.info(str(request))
    if request.method == "POST":
        notify_url = raft_config.get("my-url").strip("/") + "/notify"
        if str(request.url) == notify_url:
            body = await request.json()
            raft_config["url-map"][body["url"]] = {
                "uuid": body["uuid"],
                "status": body["status"],
                "leader": body["leader"]
            }

    if (my_url := raft_config.get("my-url")) is not None:
        reffesh_url = str(my_url).strip("/") + "/refresh"
        if str(request.url) == reffesh_url:
            await get_cluster_status()

    body = json.dumps(raft_config, indent=2, ensure_ascii=False)
    return web.Response(body=body, content_type='application/json')


async def main():
    server = web.Server(handler)
    runner = web.ServerRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', args.port)
    await site.start()
    logger.info(f"Serving on http://127.0.0.1:{args.port}/")

    await get_cluster_status()

    while True:
        num_alive_nodes = len(raft_config["url-map"]) + 1
        logger.debug(f"processing: {num_alive_nodes=}")
        await asyncio.sleep(5.0)


asyncio.run(main())
