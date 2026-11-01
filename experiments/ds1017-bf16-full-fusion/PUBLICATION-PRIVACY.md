## Publication privacy cleanup

The final public tree replaces the private LAN endpoint with a loopback example
and a site-specific Docker socket with `/run/dsv41-private/docker.sock` in six
offline helper files. Configure these examples for your own isolated environment.
The source manifest records original tested hashes and sanitized published hashes
for those helpers. All inference runtime/native source remains byte-identical to
the tested export; the live server and measured results are unchanged.

The benchmark span metadata also replaces its private endpoint with a loopback
example. Fixture text, token IDs, scores, timings and request hashes are unchanged.
Public author credits, GitHub noreply commit identity and upstream license notices
remain intentionally present for attribution. Personal home paths, private LAN
addresses, site host identifiers, credential patterns and compressed fixture
contents are checked in the final tracked publication files.
