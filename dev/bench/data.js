window.BENCHMARK_DATA = {
  "lastUpdate": 1619860662281,
  "repoUrl": "https://github.com/OpenMined/SyMPC",
  "entries": {
    "Pytest-benchmarks": [
      {
        "commit": {
          "author": {
            "email": "anubhavraj.08@gmail.com",
            "name": "Anubhav Raj Singh",
            "username": "aanurraj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7561dec407b230767b9dc517553ea77481d891e9",
          "message": "added support to generate primitives from log (#127)\n\n* added support to generate primitives from log\r\n\r\n* added tests and add get on store\r\n\r\n* fixed copy\r\n\r\n* modified logging kwargs\r\n\r\n* code imporved in generate_primitive_from_dict\r\n\r\n* more asserts to tests\r\n\r\nCo-authored-by: George-Cristian Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-04-28T19:16:54+01:00",
          "tree_id": "ba4c5e6451063185fe6166723b86d5184787ef08",
          "url": "https://github.com/OpenMined/SyMPC/commit/7561dec407b230767b9dc517553ea77481d891e9"
        },
        "date": 1619639304328,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2627793542585994,
            "unit": "iter/sec",
            "range": "stddev: 0.13450106511594326",
            "extra": "mean: 3.8054739986000072 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "spsharan2000@gmail.com",
            "name": "S P Sharan",
            "username": "Syzygianinfern0"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "08ec861c51d68c44b2facb0a1ffea7a2d824e5e6",
          "message": "Disable CUDA when using FSS (#135)\n\n* Disable GPU access rather than a hard Assertion\r\n\r\n* black\r\n\r\n* Restore environment variable value after execution\r\n\r\n* except KeyError -> .get(key, def)",
          "timestamp": "2021-05-01T10:15:25+01:00",
          "tree_id": "f1dd2d347193250a0cdfc942d8660c0c717a4e22",
          "url": "https://github.com/OpenMined/SyMPC/commit/08ec861c51d68c44b2facb0a1ffea7a2d824e5e6"
        },
        "date": 1619860661355,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.29642667203655826,
            "unit": "iter/sec",
            "range": "stddev: 0.05296507047103274",
            "extra": "mean: 3.3735155920000013 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}