# Empty root conftest: makes pytest put the repo root on sys.path (prepend
# import mode), so `import client` / `import catalog` / `import profile` work
# under any pytest invocation, not just `python -m pytest` from the root.
