import asyncio
import aiohttp
from aiohttp import web
from aiohttp.web_request import BaseRequest
import datetime
import argparse
import numpy
import yaml  # noqa 
import uuid


parser = argparse.ArgumentParser(
                    prog="Raft",
                    description="Raft Wrapper",
                    epilog='')

parser.add_argument("-p", "--port", type=int, default=9090)
args = parser.parse_args()


HEARTBEAT_TIMEOUT = 5.0    # в режиме покоя фолловер ждет сигнал от лидера
ALIVE_TIMEOUT = 1.0        # если какой-либо хост не ответил за это время - значит он умер


FOLLOWER = "FOLLOWER"
LEADER = "LEADER"
CANDIDATE = "CANDIDATE"


class Raft():
    def __init__(self):
        self.replicaset = ""
        self.id = str(uuid.uuid4())
        self.main_task: asyncio.Task | None = None
        self.semaphore = asyncio.Semaphore(0)
        self.state = FOLLOWER
        self.cluster = [
            {"host": "loacalhost", "port": 9090},
            {"host": "loacalhost", "port": 9091},
            {"host": "loacalhost", "port": 9092}
        ]

    async def pool(self, host: str, port: int):
        async with aiohttp.ClientSession() as session:
            print("pooling:", host, port)
            try:
                resp = await asyncio.wait_for(session.post(f"http://{host}:{port}/", data="{}"), HEARTBEAT_TIMEOUT)
                return (resp, None)
            except Exception as ex:
                return (None, str(ex))

    async def start_elections(self):
        res = await asyncio.gather(
            *[self.pool(h["host"], h["port"]) for h in self.cluster if h["port"] != args.port]
        )
        for r in res:
            print("RR> ", r)

    async def wait_heartbeat(self):
        s = datetime.datetime.now()
        await asyncio.wait_for(self.semaphore.acquire(), timeout=HEARTBEAT_TIMEOUT)
        print(datetime.datetime.now() - s)

    async def heartbeat(self, data) -> None:
        if self.state == FOLLOWER:
            self.semaphore.release()


raft = Raft()


async def handler(request: BaseRequest):
    print(request.url)
    await raft.heartbeat(None)
    return web.Response(text=f"OK + {request.url}")


async def main():
    server = web.Server(handler)
    runner = web.ServerRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', args.port)
    await site.start()
    print(f"Serving on http://127.0.0.1:{args.port}/")

    while True:
        start = datetime.datetime.now()
        if raft.state == FOLLOWER:
            # последователь ждет хардбиты и/или данные
            try:
                print(f"{datetime.datetime.now()}: create raft and await")
                await raft.wait_heartbeat()
            except Exception as ex:
                print("Переключаемся: ", datetime.datetime.now() - start, ex)
                raft.state = CANDIDATE
        elif raft.state == CANDIDATE:
            print("I'm a candidate, start elections...")
            await asyncio.sleep(numpy.random.random())
            await raft.start_elections()

        else:  # LEADER
            print("I'm the leader")
            await asyncio.sleep(numpy.random.random())


asyncio.run(main())
