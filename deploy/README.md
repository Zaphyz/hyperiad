# Hyperiad's public server

A Cloudflare Worker (`src/index.ts`) in front of Cloudflare Containers running this repository's
`hyperiad serve --isolate` (`Dockerfile`, built from the repository root). It answers at
`https://arity.science/api/hyperiad/` (and at the Worker's `workers.dev` address):

| Request | Answered by |
|---|---|
| `GET /info` | the Worker: the server's budget and limits, without waking a container |
| `POST /run` `{code, replay}` | a container: replays the earlier cells, then runs the cell |

## Limits

Sized for low traffic from anonymous visitors. Change them in `src/index.ts` (`LIMITS`, `INSTANCES`, `RATE`)
and `wrangler.jsonc` (`max_instances`, `instance_type`).

- **Cost estimate**: a job whose estimate is over 2e9 operations (the browser's limit is 2e7) is refused
  before it runs, and the notebook shows the "too heavy to be run here" message.
- **Per job**: 60 s of wall-clock time and 1 GiB of memory, in a child process of its own.
- **Concurrency**: 2 container instances (1/2 vCPU, 4 GiB each), one job at a time each. When both are
  busy the visitor is told to try again in a minute. An idle instance sleeps after 5 minutes.
- **Rate**: 4 jobs a minute and 30 an hour per visitor (by IP address), 30 a minute overall. One
  Durable Object (`Gate`) counts every request, so the limits are exact; Cloudflare's rate limiting
  binding was tried first and let bursts of 40 through.
- **Request size**: 200 kB.

## Deploying

To run your own server from a fork, change `name` and `routes` in `wrangler.jsonc` (the route here is
arity.science's; remove `routes` to use the Worker's `workers.dev` address), and add your site to
`ALLOWED_ORIGINS` in `src/index.ts`.

Workers Builds deploys this folder on every push to `main`: in the Cloudflare dashboard,
Workers & Pages → Create → Import a repository → `Zaphyz/hyperiad`, root directory
`deploy`, deploy command `npx wrangler deploy`. The first deploy builds and uploads the image;
later pushes rebuild it when the Python changes.

To check after a deploy:

```bash
curl https://arity.science/api/hyperiad/info
curl -X POST https://arity.science/api/hyperiad/run -H 'Content-Type: application/json' -d '{"code": "a + b", "replay": ["a = 2", "b = 3"]}'
```

The first `run` after a quiet spell waits a few seconds for a container to start.

## Testing locally

```bash
docker build -f deploy/Dockerfile -t hyperiad-server .     # from the repository root
docker run --rm -p 8080:8080 --memory 4g --cpus 0.5 hyperiad-server
curl -X POST localhost:8080/api/run -H 'Content-Type: application/json' -d '{"code": "1 + 1", "replay": []}'
```

`npm run check` type-checks the Worker.
