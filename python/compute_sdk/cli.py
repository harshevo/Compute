import argparse
import json
from pathlib import Path

from .client import Cluster


def main():
    parser = argparse.ArgumentParser(prog="compute")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices")
    run = sub.add_parser("run")
    run.add_argument("--node", default="auto")
    run.add_argument("--device", default="auto")
    run.add_argument("--data")
    run.add_argument("--timeout", type=float, default=86400)
    run.add_argument("script")
    run.add_argument("args", nargs=argparse.REMAINDER)
    for name in ("logs", "status", "cancel", "resume", "pull"):
        p = sub.add_parser(name)
        p.add_argument("--node", required=True)
        if name == "logs":
            p.add_argument("-f", "--follow", action="store_true")
        p.add_argument("id")
        if name == "pull":
            p.add_argument("destination", nargs="?", default="./results")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("configuration", type=Path)
    pool = sub.add_parser("pool")
    pool.add_argument("--nodes", nargs="+", required=True)
    pool.add_argument("--strategy", choices=["local_sgd", "lora_avg"], default="local_sgd")
    pool.add_argument("--rounds", type=int, default=10)
    pool.add_argument("--sync-every", type=int, default=10)
    pool.add_argument("--batch-size", type=int, default=8)
    pool.add_argument("--device", default="auto")
    pool.add_argument("--results", default="pool-results")
    pool.add_argument("script")
    gateway = sub.add_parser("gateway")
    gateway.add_argument("--node", action="append", required=True)
    gateway.add_argument("--device", default="auto")
    gateway.add_argument("--port", type=int, default=8080)
    gateway.add_argument("script")
    args = parser.parse_args()
    cluster = Cluster()
    if args.command == "devices":
        print(json.dumps(cluster.devices(), indent=2))
    elif args.command == "run":
        trailing = args.args[1:] if args.args[:1] == ["--"] else args.args
        job = cluster.submit(args.script, args.node, args.data, trailing, args.device, args.timeout)
        print(json.dumps({"id": job.id, "node": job.node}))
    elif args.command in {"logs", "status", "cancel", "resume", "pull"}:
        job = cluster.job(args.node, args.id)
        if args.command == "logs":
            job.stream_logs(args.follow)
        elif args.command == "pull":
            print(job.download(args.destination))
        else:
            print(json.dumps(getattr(job, args.command)()))
    elif args.command == "sweep":
        import tomllib
        config = tomllib.loads(args.configuration.read_text())
        script = (args.configuration.parent / config["script"]).resolve()
        jobs = cluster.sweep(script, config["trials"], config.get("nodes"), config.get("device", "auto"))
        print(json.dumps([{"id": job.id, "node": job.node} for job in jobs]))
    elif args.command == "pool":
        from .pool import PooledTrainer
        trainer = PooledTrainer(args.script, args.nodes, cluster, args.strategy, args.device,
                                args.batch_size, args.sync_every, args.results)
        trainer.fit(args.rounds)
        print(json.dumps({"results": str(trainer.results), "rounds": len(trainer.history)}))
    elif args.command == "gateway":
        from .inference import RemoteModel, create_gateway
        import uvicorn
        models = [RemoteModel(args.script, node, cluster, args.device) for node in args.node]
        try:
            uvicorn.run(create_gateway(models), host="127.0.0.1", port=args.port)
        finally:
            for model in models:
                model.close()


if __name__ == "__main__":
    main()
