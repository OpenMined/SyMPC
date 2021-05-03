window.BENCHMARK_DATA = {
  "lastUpdate": 1620034473408,
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
      },
      {
        "commit": {
          "author": {
            "email": "danielorihuela@users.noreply.github.com",
            "name": "danielorihuela",
            "username": "danielorihuela"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "728b978651d5a6bbeb75bdb83dee0ad5e73fd957",
          "message": "Style/flake8 configuration (#132)\n\n* refactor: all flake8 configuration in one place\r\n\r\nWe copied the flake8 configuration of pysyft\r\nand added some options.\r\n\r\nAdded:\r\n\r\n- F401 since in Pysyft you can find a lot of\r\n`# noqa 401` through the code.\r\n- DAR101 and DAR201 to ignore only on tests.\r\n  Seems fair, since arguments and returns should be already commented on\r\n  the source code.\r\n  Also, it could clatter the code explaining fixtures.\r\n- Exclude rst, txt, md files and the setup.cfg, since they are not\r\nsource code.\r\n\r\n* fix: flake8 warnings\r\n\r\n* chore: typo\r\n\r\n* fix: style warnings\r\n\r\n* chore: remove _ on methods returning single value\r\n\r\n* chore: update black line length option",
          "timestamp": "2021-05-01T14:19:24+01:00",
          "tree_id": "06aa261ccb6ef35eb114fbf5b6b76ebb3aca971a",
          "url": "https://github.com/OpenMined/SyMPC/commit/728b978651d5a6bbeb75bdb83dee0ad5e73fd957"
        },
        "date": 1619875328679,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.22765216072718536,
            "unit": "iter/sec",
            "range": "stddev: 0.1391600686772205",
            "extra": "mean: 4.392666411799991 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "kamathhrishi@gmail.com",
            "name": "Hrishikesh Kamath",
            "username": "kamathhrishi"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "95e351db48172fd2df0d89383a3023cb747306da",
          "message": "Separate tests for exceptions  (#122)\n\n* Initialize exeception tests\r\n\r\n* Add no session exception test\r\n\r\n* Run precommit hook\r\n\r\n* Test to ensure exception is thrown for invalid protocol\r\n\r\n* Test exception for div on MPC Tensor\r\n\r\n* precommit hook\r\n\r\n* Add accidental function call in tests\r\n\r\n* Precoommit hook\r\n\r\n* Add exception test for conv2d kernel mismatch\r\n\r\n* Rename test functions\r\n\r\n* Test exception for invalid op\r\n\r\n* Add test for exception for dividing sharetensor with float\r\n\r\n* Add exception test for invalid ring size\r\n\r\n* Add exception test for unsupported operation on FSS\r\n\r\n* Add exception for Insufficient FSS primitives\r\n\r\n* modify div with float exception test to use get_clients functioon\r\n\r\n* Exception tests use get_primitive()\r\n\r\n* Ran precommit\r\n\r\n* Precommit hook\r\n\r\n* Modify div_float exception test to remove redundant step\r\n\r\n* Modify test name to setup_mpc\r\n\r\n* Removed redundant test\r\n\r\n* Remove redundant statements in test_mpc_share_nosession_exception and remove uncessary get_clients()\r\n\r\n* Change variable name in no session exception test\r\n\r\n* Modify test name and remove unused parameters\r\n\r\n* Rechange function name\r\n\r\n* Ran precommit hook\r\n\r\n* Remove unused variables",
          "timestamp": "2021-05-03T10:32:03+01:00",
          "tree_id": "5336ac70feaa912c55504e156540bc2a9505395a",
          "url": "https://github.com/OpenMined/SyMPC/commit/95e351db48172fd2df0d89383a3023cb747306da"
        },
        "date": 1620034472399,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2622847872635144,
            "unit": "iter/sec",
            "range": "stddev: 0.1042745907220382",
            "extra": "mean: 3.8126496410000015 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}