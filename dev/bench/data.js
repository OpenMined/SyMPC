window.BENCHMARK_DATA = {
  "lastUpdate": 1622883228087,
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
      },
      {
        "commit": {
          "author": {
            "email": "16245436+jmaunon@users.noreply.github.com",
            "name": "jmaunon",
            "username": "jmaunon"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c47d1cb4ace446de0a295e1b8ddab1041f9d7c91",
          "message": "Add auto-flake (#139)\n\n* Add auto-flake\r\n\r\n* Add --recursive and update GA test\r\n\r\n* Add foo file that will be deleted after testing\r\n\r\n* Delete foo.py\r\n\r\n* Add all-unused-imports\r\n\r\nCo-authored-by: George-Cristian Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-05-03T19:39:42+01:00",
          "tree_id": "4365619d6123f75727e1b3db503427c0edd4d833",
          "url": "https://github.com/OpenMined/SyMPC/commit/c47d1cb4ace446de0a295e1b8ddab1041f9d7c91"
        },
        "date": 1620067331833,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.24349522707325272,
            "unit": "iter/sec",
            "range": "stddev: 0.10109338241279515",
            "extra": "mean: 4.106856680599992 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72750519f1a19a0bf66d2b930241ffe9fd39ffd6",
          "message": "Add gradient functions (#138)\n\n* Test\r\n\r\n* Add stub backprop\r\n\r\n* Move backward to MPCTensor\r\n\r\n* Change pow to mpc_pow\r\n\r\n* Add documentation for functions\r\n\r\n* Add tests",
          "timestamp": "2021-05-05T20:32:12+01:00",
          "tree_id": "d96190e70942f75d2c259c847db385af1cc6c8c6",
          "url": "https://github.com/OpenMined/SyMPC/commit/72750519f1a19a0bf66d2b930241ffe9fd39ffd6"
        },
        "date": 1620243262878,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2967479196897495,
            "unit": "iter/sec",
            "range": "stddev: 0.11495999713714863",
            "extra": "mean: 3.3698635563999972 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "4b333154dfbdfd56063e7807cda4506acd8aa76e",
          "message": "support to use pre generated primitives (#141)\n\n* added conditional genration for generating primitive\r\n\r\n* fixes\r\n\r\n* fixed except block\r\n\r\n* added custom exceptions\r\n\r\n* added tests\r\n\r\n* moved exception file\r\n\r\n* more specific exceptions\r\n\r\n* parameterized tests",
          "timestamp": "2021-05-06T19:19:00+01:00",
          "tree_id": "57af9b4c8e561abc3d99269521abec5d316fcc33",
          "url": "https://github.com/OpenMined/SyMPC/commit/4b333154dfbdfd56063e7807cda4506acd8aa76e"
        },
        "date": 1620325280889,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.25876093035228687,
            "unit": "iter/sec",
            "range": "stddev: 0.13457891608113143",
            "extra": "mean: 3.8645710487999962 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "002a03356313b8ddd73889555ee874adef8f0b55",
          "message": "added support for private division (#143)\n\n* added support for private div\r\n\r\n* fixed tests\r\n\r\n* converted div to truediv",
          "timestamp": "2021-05-09T20:47:25+01:00",
          "tree_id": "7263fa7cb915f8bf1c47864c95037d2f417f8b37",
          "url": "https://github.com/OpenMined/SyMPC/commit/002a03356313b8ddd73889555ee874adef8f0b55"
        },
        "date": 1620589792573,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.24822550201337962,
            "unit": "iter/sec",
            "range": "stddev: 0.11549286389576566",
            "extra": "mean: 4.028594934400007 sec\nrounds: 5"
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
          "id": "16264701b3ee6787cf40edd0ffeeda4c589035ad",
          "message": "Sort list of MPC Tensors (#140)\n\n* Initiate sorting list of MPC Tensors\r\n\r\n* Precommit hook\r\n\r\n* Rename test file\r\n\r\n* Add documentation and type annotations\r\n\r\n* Modify documentation\r\n\r\n* Modify doc strring in __init__\r\n\r\n* Rename sort function\r\n\r\n* Update __init__.py\r\n\r\n* Update __init__.py\r\n\r\n* Raise ValueError due to invalid list size and  test for exception\r\n\r\n* Raise ValueError due to invalid list size and  test for exception\r\n\r\n* Added secure implementation of bubblesort\r\n\r\n* Change name from applications to algorithms\r\n\r\n* Fix tests\r\n\r\n* Style  changes\r\n\r\n* Remove applications folder and fix darlint issue\r\n\r\n* Modified docstring\r\n\r\n* Renamed files\r\n\r\n* Parameterized pytest test and small style changes\r\n\r\n* Remove unnecessary new lines\r\n\r\n* save 1-check to improve performance\r\n\r\n* Precommit hook\r\n\r\nCo-authored-by: George-Cristian Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-05-09T21:15:54+01:00",
          "tree_id": "f48bb505332be7275f8a38dff5f361d25320be36",
          "url": "https://github.com/OpenMined/SyMPC/commit/16264701b3ee6787cf40edd0ffeeda4c589035ad"
        },
        "date": 1620591504551,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2499550858080884,
            "unit": "iter/sec",
            "range": "stddev: 0.10447490394441059",
            "extra": "mean: 4.000718756200001 sec\nrounds: 5"
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
          "id": "bf3e11a386e23b4cbc46e437f6cea4f10ff46344",
          "message": "feat: move allowlist from syft to sympc (#162)\n\n* feat: move allowlist from syft to sympc\r\n\r\n* refactor: move allowlist to share tensor file\r\n\r\n* feat: move allowlists from pysyft to sympc\r\n\r\n* refactor: allowlist to api.py\r\n\r\n* docs: api.py\r\n\r\n* chore: improve import section",
          "timestamp": "2021-05-13T00:04:42+01:00",
          "tree_id": "b88f068b836667e5ae768c2b27b105eb343f5c27",
          "url": "https://github.com/OpenMined/SyMPC/commit/bf3e11a386e23b4cbc46e437f6cea4f10ff46344"
        },
        "date": 1620860823025,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2582252508254092,
            "unit": "iter/sec",
            "range": "stddev: 0.1022160260641643",
            "extra": "mean: 3.872587970399991 sec\nrounds: 5"
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
          "id": "1a9ebab9cc61b6e43a06908a1b6055e32b3743d7",
          "message": "fix: add lacking methods (#170)\n\n* fix: add lacking methods\r\n\r\n* chore: add noqa",
          "timestamp": "2021-05-13T22:26:25+01:00",
          "tree_id": "5263909492909421e7e1d52ef8d16ea5f1109c2d",
          "url": "https://github.com/OpenMined/SyMPC/commit/1a9ebab9cc61b6e43a06908a1b6055e32b3743d7"
        },
        "date": 1620941326817,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.25872413400974914,
            "unit": "iter/sec",
            "range": "stddev: 0.210549758410243",
            "extra": "mean: 3.865120676999999 sec\nrounds: 5"
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
          "id": "1d5f6abbbb9e90dd94c6de31105581ea7b261ae4",
          "message": "Initialise Replicated Shared Tensor  (#167)\n\n* initialize RSTensor\r\n\r\n* Update docstrings\r\n\r\n* Update docstrings\r\n\r\n* Make suggested changes",
          "timestamp": "2021-05-16T21:13:01+01:00",
          "tree_id": "cce555b8bd40f39cdb248f523e3695d38eebc2a0",
          "url": "https://github.com/OpenMined/SyMPC/commit/1d5f6abbbb9e90dd94c6de31105581ea7b261ae4"
        },
        "date": 1621196147878,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.24575550833684484,
            "unit": "iter/sec",
            "range": "stddev: 0.11819067720899928",
            "extra": "mean: 4.0690847857999985 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "63cf891cd91634b4c049dea12d9c28eece7d60ad",
          "message": "Add training POC (#161)",
          "timestamp": "2021-05-18T00:22:21+01:00",
          "tree_id": "ca5e3198b99a3f21e748d58d2c47fda7ed9be9dd",
          "url": "https://github.com/OpenMined/SyMPC/commit/63cf891cd91634b4c049dea12d9c28eece7d60ad"
        },
        "date": 1621293902065,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.25486938984758883,
            "unit": "iter/sec",
            "range": "stddev: 0.11921164140953354",
            "extra": "mean: 3.9235782711999945 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lihu723@gmail.com",
            "name": "libra",
            "username": "libratiger"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f9f0ef9b5ecfbf16df1b014e2ee57c12fb1dc42",
          "message": "Add the mean reduction for mse_loss (#180)\n\n* Add the `mean` reduction method for `mse_loss`\r\n\r\nAdd the `mean` reduction method for `mse_loss`, try to keep the same with pytorch\r\n\r\n* Fix the mse_loss\r\n\r\n* fix the mse_loss",
          "timestamp": "2021-05-21T19:30:13+05:30",
          "tree_id": "55b871e1b57873399bfb34aea974b9da78beaf9f",
          "url": "https://github.com/OpenMined/SyMPC/commit/3f9f0ef9b5ecfbf16df1b014e2ee57c12fb1dc42"
        },
        "date": 1621605783906,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2327509049060636,
            "unit": "iter/sec",
            "range": "stddev: 0.15897235727383152",
            "extra": "mean: 4.296438720200001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5fe901a30744525caafcefa714f2c71f2fd02247",
          "message": "POC sharing at protocol level (#178)\n\n* POC sharing at protocol level\r\n\r\n* Fix comments\r\n\r\n* Add tests",
          "timestamp": "2021-05-21T20:11:08+01:00",
          "tree_id": "22b6f159cf1a62c7b5b48022d3a6b5c6b7f7ffee",
          "url": "https://github.com/OpenMined/SyMPC/commit/5fe901a30744525caafcefa714f2c71f2fd02247"
        },
        "date": 1621624431586,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2386627296746943,
            "unit": "iter/sec",
            "range": "stddev: 0.13097348566082334",
            "extra": "mean: 4.19001325160001 sec\nrounds: 5"
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
          "id": "7875f1692c1dbcc3256f6b548981fb9113aa3767",
          "message": "Init RS Tensor (#186)\n\n* Modify SyMPC PySyft API file for RSTensor\r\n\r\n* Add RSTensor to __init__.py\r\n\r\n* Fix wrong import\r\n\r\n* Init RSTensor\r\n\r\n* Precommit hook\r\n\r\n* Add a RSTensor test\r\n\r\n* Add a test\r\n\r\n* Add import test\r\n\r\n* Modify hook methods\r\n\r\n* Modify API.py\r\n\r\n* Precommit hook",
          "timestamp": "2021-05-22T21:12:43+01:00",
          "tree_id": "88bde98b62ca1be8b65ac971fac40963f071361d",
          "url": "https://github.com/OpenMined/SyMPC/commit/7875f1692c1dbcc3256f6b548981fb9113aa3767"
        },
        "date": 1621714536511,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.22368207435308823,
            "unit": "iter/sec",
            "range": "stddev: 0.17210085472287695",
            "extra": "mean: 4.470630929599986 sec\nrounds: 5"
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
          "id": "80250b1f951667bb19e1b181dab249a337850ae2",
          "message": "docs: contributing (#174)\n\n* docs: contributing\r\n\r\nCopied and modified from PySyft contributing.\r\n\r\n* docs: escape symbol\r\n\r\n* fix: docs link\r\n\r\n* docs: change python 3.6 to python 3.7\r\n\r\n* docs: change some examples, steps and commands\r\n\r\n* docs: improve explanations\r\n\r\n* docs: remove wsl based systems\r\n\r\n* build: macos and windows python tests\r\n\r\n* docs: how to install sphinx\r\n\r\nCo-authored-by: George-Cristian Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-05-22T22:16:18+01:00",
          "tree_id": "647dc834316a9f00d92b3521bedae302dcfdcab8",
          "url": "https://github.com/OpenMined/SyMPC/commit/80250b1f951667bb19e1b181dab249a337850ae2"
        },
        "date": 1621718346041,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.24308014331799532,
            "unit": "iter/sec",
            "range": "stddev: 0.12393075634008337",
            "extra": "mean: 4.113869550799995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "murarugeorgec@gmail.com",
            "name": "George Muraru",
            "username": "gmuraru"
          },
          "distinct": true,
          "id": "3583cc9d89449dc2d5c7fed020e89e1a0b9384f4",
          "message": "Change name for benchmark jobs",
          "timestamp": "2021-05-23T11:28:54+01:00",
          "tree_id": "2a396f59bbd80aa874ce08d9da0acf4ed870a0bd",
          "url": "https://github.com/OpenMined/SyMPC/commit/3583cc9d89449dc2d5c7fed020e89e1a0b9384f4"
        },
        "date": 1621765893554,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.26269250711508135,
            "unit": "iter/sec",
            "range": "stddev: 0.09797851708885029",
            "extra": "mean: 3.8067321028000096 sec\nrounds: 5"
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
          "id": "be0f3b09117b0988f6fa1b68d771084fb4c00ed3",
          "message": "Add RSTensor SyMPC API  (#185)\n\n* Modify SyMPC PySyft API file for RSTensor\r\n\r\n* Add RSTensor to __init__.py\r\n\r\n* Fix wrong import\r\n\r\n* Modify API to include ReplicatedSharedTensor\r\n\r\n* Update replicatedshare_tensor.py\r\n\r\n* Precommit hook\r\n\r\nCo-authored-by: George-Cristian Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-05-23T13:14:29+01:00",
          "tree_id": "38d1abe2a0c148e0a261f80615faa387efc30664",
          "url": "https://github.com/OpenMined/SyMPC/commit/be0f3b09117b0988f6fa1b68d771084fb4c00ed3"
        },
        "date": 1621772224837,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2425339594370461,
            "unit": "iter/sec",
            "range": "stddev: 0.15939222600030367",
            "extra": "mean: 4.123133940999992 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43314053+rasswanth-s@users.noreply.github.com",
            "name": "rasswanth",
            "username": "rasswanth-s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c7ddbdc28ec115ac51fdef14ffde276955512c35",
          "message": "Initialize FALCON Protocol (#189)\n\n* Intialize FALCON Protocol\r\n\r\n* Added test for session\r\n\r\n* Modified Protocol name convention\r\n\r\n* Modified worklows Parallel Execution",
          "timestamp": "2021-05-25T19:49:17+05:30",
          "tree_id": "e51a6aa89271afc5c2c2df9b1e3de8f3d86f7e14",
          "url": "https://github.com/OpenMined/SyMPC/commit/c7ddbdc28ec115ac51fdef14ffde276955512c35"
        },
        "date": 1621952539451,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_conv_model",
            "value": 0.2322375686139575,
            "unit": "iter/sec",
            "range": "stddev: 0.10087103494405986",
            "extra": "mean: 4.305935538199998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lihu723@gmail.com",
            "name": "libra",
            "username": "libratiger"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fcd7a40f066cf9b3278ac78013b3c24d4fb5436c",
          "message": "Add the benchmkar for the inference (#191)",
          "timestamp": "2021-05-26T16:46:19+05:30",
          "tree_id": "1194475e020762341c4bd307f617a2907a41f8fa",
          "url": "https://github.com/OpenMined/SyMPC/commit/fcd7a40f066cf9b3278ac78013b3c24d4fb5436c"
        },
        "date": 1622027941710,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.32250782570318476,
            "unit": "iter/sec",
            "range": "stddev: 0.06947869721977652",
            "extra": "mean: 3.100699953000009 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "15707966e3ff4a3b919968744698b828f79c1932",
          "message": "Grad function Conv2d (#179)\n\n* conv_transpose2d\r\n\r\n* added conv_trans tests\r\n\r\n* fixes\r\n\r\n* added grad conv2d for 2 parties\r\n\r\n* fixed imports\r\n\r\n* fixed bugs and added tests\r\n\r\n* black format\r\n\r\n* added tests for get_input_padding",
          "timestamp": "2021-05-26T18:40:01+01:00",
          "tree_id": "4c21db13a76b71a4f77e1972efc96b79b948d70f",
          "url": "https://github.com/OpenMined/SyMPC/commit/15707966e3ff4a3b919968744698b828f79c1932"
        },
        "date": 1622050964803,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.3231781607976062,
            "unit": "iter/sec",
            "range": "stddev: 0.021635432999829892",
            "extra": "mean: 3.0942684912000002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "36797859+Param-29@users.noreply.github.com",
            "name": "Param Mirani",
            "username": "Param-29"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6437f0456c9fc70434b45a1457f441b81827dcf3",
          "message": "safety, synk added to CI (#166)\n\n* safety added to CI\r\n\r\n* adding synk:\r\n\r\nIt will run after code is merged as it requires a secret token(to be set up yet..)\r\n\r\n* Making suggested changes\r\n: test names reverted\r\n: moved git-PySyft to the end of the file\r\n\r\n* Making suggested changes\r\n: thanks, aanurraj\r\n\r\n* Change with SNYK secrets\r\n\r\n* Change to push on main\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-05-29T13:28:26+05:30",
          "tree_id": "2d8786e06630eaffc7a9a2ac6d62550507a47777",
          "url": "https://github.com/OpenMined/SyMPC/commit/6437f0456c9fc70434b45a1457f441b81827dcf3"
        },
        "date": 1622275679960,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.3079287760341951,
            "unit": "iter/sec",
            "range": "stddev: 0.011573617402594999",
            "extra": "mean: 3.247504221200006 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e6fc61af5610cb6d4b5d162c4c2a2fae61477de0",
          "message": "Revert \"safety, synk added to CI (#166)\" (#200)\n\nThis reverts commit 6437f0456c9fc70434b45a1457f441b81827dcf3.",
          "timestamp": "2021-05-29T12:15:16+01:00",
          "tree_id": "4c21db13a76b71a4f77e1972efc96b79b948d70f",
          "url": "https://github.com/OpenMined/SyMPC/commit/e6fc61af5610cb6d4b5d162c4c2a2fae61477de0"
        },
        "date": 1622287070490,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.32420852608159506,
            "unit": "iter/sec",
            "range": "stddev: 0.07769171378023677",
            "extra": "mean: 3.0844346140000196 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43314053+rasswanth-s@users.noreply.github.com",
            "name": "rasswanth",
            "username": "rasswanth-s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "622ee4d8d9a7559e8c4ad412157dd12024eab12e",
          "message": "Modified NOT_COMPARE attribute in Session class. (#203)\n\n* Intialize FALCON Protocol\r\n\r\n* Added test for session\r\n\r\n* Modified Protocol name convention\r\n\r\n* Modified worklows Parallel Execution\r\n\r\n* Modified Session.NOT_COMPARE Attribute\r\n\r\n* Refactored ShareTensor tests",
          "timestamp": "2021-05-30T18:20:43+01:00",
          "tree_id": "87e56ef0a4a416d42da8a92fc3a62370a32db18f",
          "url": "https://github.com/OpenMined/SyMPC/commit/622ee4d8d9a7559e8c4ad412157dd12024eab12e"
        },
        "date": 1622395422392,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.2868919855117245,
            "unit": "iter/sec",
            "range": "stddev: 0.02921424735613735",
            "extra": "mean: 3.4856324000000085 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "e653498a602700d9bc0715230379ffb007a289d4",
          "message": "added GradFlatten Tests (#204)",
          "timestamp": "2021-05-30T18:21:54+01:00",
          "tree_id": "7ce84fa6368cc0db53640a9b9ef165475087aeb1",
          "url": "https://github.com/OpenMined/SyMPC/commit/e653498a602700d9bc0715230379ffb007a289d4"
        },
        "date": 1622396997656,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.36368413808403444,
            "unit": "iter/sec",
            "range": "stddev: 0.06104047818001348",
            "extra": "mean: 2.749638753199997 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "0ba4feab9c55fbfed9c3d7c734a489a8ac3a2ef2",
          "message": "added support for  GradReshape (#198)\n\n* added GradReshape\r\n\r\n* test improvement\r\n\r\n* resolved conflicts",
          "timestamp": "2021-05-30T19:40:41+01:00",
          "tree_id": "68bae76ed9178ef8a66df16f3e3bba6b875e1f41",
          "url": "https://github.com/OpenMined/SyMPC/commit/0ba4feab9c55fbfed9c3d7c734a489a8ac3a2ef2"
        },
        "date": 1622400180365,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.34577954651084963,
            "unit": "iter/sec",
            "range": "stddev: 0.0187890722762834",
            "extra": "mean: 2.8920160550000107 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "36797859+Param-29@users.noreply.github.com",
            "name": "Param Mirani",
            "username": "Param-29"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a09246a11997ed1fe440e3ecb05105dd6353f2d6",
          "message": "Re: Synk and security ci (#208)\n\n* safety added to CI\r\n\r\n* adding synk:\r\n\r\nIt will run after code is merged as it requires a secret token(to be set up yet..)\r\n\r\n* Making suggested changes\r\n: test names reverted\r\n: moved git-PySyft to the end of the file\r\n\r\n* Making suggested changes\r\n: thanks, aanurraj\r\n\r\n* Change with SNYK secrets\r\n\r\n* Change to push on main\r\n\r\n* token set for synk\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-06-01T14:29:29+01:00",
          "tree_id": "5789315ac058ad69999d2dac2f67c73471e8652e",
          "url": "https://github.com/OpenMined/SyMPC/commit/a09246a11997ed1fe440e3ecb05105dd6353f2d6"
        },
        "date": 1622554340362,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.3050906827177713,
            "unit": "iter/sec",
            "range": "stddev: 0.008188707328404507",
            "extra": "mean: 3.277713993400005 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "78d8539e1a923e09f0f8a4a3aee3a7c8293cca0e",
          "message": "added  codecov (#205)\n\n* added  codecov\r\n\r\n* test new Job\r\n\r\n* test new Job\r\n\r\n* test new Job\r\n\r\n* test new Job\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* fixes\r\n\r\n* added secret key",
          "timestamp": "2021-06-01T19:40:40+01:00",
          "tree_id": "a3bea155ae472fbd413556b77afc77391953aaa3",
          "url": "https://github.com/OpenMined/SyMPC/commit/78d8539e1a923e09f0f8a4a3aee3a7c8293cca0e"
        },
        "date": 1622572993225,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.320821345598861,
            "unit": "iter/sec",
            "range": "stddev: 0.026906499078683088",
            "extra": "mean: 3.116999581600004 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43314053+rasswanth-s@users.noreply.github.com",
            "name": "rasswanth",
            "username": "rasswanth-s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "479e825d787176678f8d64b70a75ec91e7a8c4b7",
          "message": "Added Hook method and property -RSTensor (#194)\n\n* Added Hook method and property - RSTensor\r\n\r\n* Modified RSTensor to torch.Tensor\r\n\r\n* Modified loop parameters,docstring,testcase\r\n\r\n* Refactored and added more tests for RSTensor",
          "timestamp": "2021-06-02T20:57:04+05:30",
          "tree_id": "6fba25e51c2cf6abd671753fc8de7a31319c6fba",
          "url": "https://github.com/OpenMined/SyMPC/commit/479e825d787176678f8d64b70a75ec91e7a8c4b7"
        },
        "date": 1622647766881,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.3398405984157283,
            "unit": "iter/sec",
            "range": "stddev: 0.023756125168244456",
            "extra": "mean: 2.942556023800006 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9abaa29e83adb6aef3d316a9215dec2eb2a571e3",
          "message": "Use Session UUID for retrieving session in parties (#190)\n\n* - add get_session function to utils.py\r\n- edit parallel_execution functions to remove session from args, and use get_session()\r\n- remove import sympc.session.Session (not used)\r\n\r\n* add import of get_session to utils __init__.py\r\n\r\n* Edit formatting\r\n\r\n* remove Session import from fss\r\n\r\n* add return typing for get_session()\r\n\r\n* move get_session to session.py\r\n\r\n* try setting current_session as library variable\r\n\r\n* return sympc.session.current_session in deserialize step\r\n\r\n* Use uuid session\r\n\r\n* Fix tests\r\n\r\n* Add try catch block back\r\n\r\n* Add tests back and fix comment\r\n\r\n* Remove snyk until we get token\r\n\r\nCo-authored-by: Lina <lina.mntran@gmail.com>\r\nCo-authored-by: Lina Tran <10761918+linamnt@users.noreply.github.com>",
          "timestamp": "2021-06-03T12:17:09+01:00",
          "tree_id": "8d7b6e67fe4323914fdcae40c86c0649128a20e2",
          "url": "https://github.com/OpenMined/SyMPC/commit/9abaa29e83adb6aef3d316a9215dec2eb2a571e3"
        },
        "date": 1622723394135,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1247825814723139,
            "unit": "iter/sec",
            "range": "stddev: 0.05124760358301393",
            "extra": "mean: 8.013939030599994 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43314053+rasswanth-s@users.noreply.github.com",
            "name": "rasswanth",
            "username": "rasswanth-s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7a988fd64e8e026602c5cae5bd122bad958ac415",
          "message": "Fixed Point Encoding - RSTensor (#202)\n\n* Intialize FALCON Protocol\r\n\r\n* Added test for session\r\n\r\n* Modified Protocol name convention\r\n\r\n* Modified worklows Parallel Execution\r\n\r\n* Fixed Point Encoding- RSTensor\r\n\r\n* Modified RSTensor to use Session UUID\r\n\r\n* Modified Tests\r\n\r\n* Added test for fixed point\r\n\r\n* Modified Docstring",
          "timestamp": "2021-06-04T11:48:08+05:30",
          "tree_id": "c581f78b6a6c7d04f3f3510ce9a3a0206258ff08",
          "url": "https://github.com/OpenMined/SyMPC/commit/7a988fd64e8e026602c5cae5bd122bad958ac415"
        },
        "date": 1622787663542,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12322398659018322,
            "unit": "iter/sec",
            "range": "stddev: 0.0666355959153346",
            "extra": "mean: 8.115303096999998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43314053+rasswanth-s@users.noreply.github.com",
            "name": "rasswanth",
            "username": "rasswanth-s"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "05f8f8063826a27e07f0b7b906c23cd8711320c2",
          "message": "Refactored hook method and property -RSTensor (#213)",
          "timestamp": "2021-06-04T20:35:36+05:30",
          "tree_id": "e566b55f1a30a012c4b9067346dd3c81d8a5b049",
          "url": "https://github.com/OpenMined/SyMPC/commit/05f8f8063826a27e07f0b7b906c23cd8711320c2"
        },
        "date": 1622821697423,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.10500642858386341,
            "unit": "iter/sec",
            "range": "stddev: 0.044129819726946164",
            "extra": "mean: 9.523226468 sec\nrounds: 5"
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
          "id": "6b17043088aac9939957ce038eb49d40b6fd302f",
          "message": "Add Sigmoid (cheby), Static Methods (stack, cat) (#136)\n\n* Add chebyshev algo\r\n\r\n* Add chebyshev test\r\n\r\n* Fix for negative elements\r\nFor some reason, they are lower than expected result.\r\nHence, we make them positive, compute and later return 1-Ans.\r\n\r\n* Add Reference Paper of Implementation\r\n\r\n* Remove a double sign inversion for negatives\r\n\r\n* tensor_8 -> positive_tensor_rescaled\r\n\r\n* Make operation less accurate but more reliable\r\nThe method is flaky due to computation of powers of 12. Reducing them for reliability.\r\n\r\n* Crypten's version of `sigmoid` chebyshev approx\r\n\r\n* Add `squeeze`, `cat`, and `stack` methods\r\n\r\n* WIP Code\r\n\r\n* Fix\r\n\r\n* Crypten implementation complete\r\n\r\n* Sweet spot for implementations\r\n\r\n* Implement `cat` method\r\n\r\n* Docstrings for static\r\n\r\n* Empty commit\r\nFor some reason, git is not pushing xD\r\n\r\n* cheby->cheby-aliter, cheby-crypten->cheby\r\nSome more docstrings\r\n\r\n* Fix test method name\r\n\r\n* Fix `parallel_execution` args\r\nThis is due to a new PR being merged\r\nFix method name in tanh\r\n\r\n* Tests for `static.py`\r\n\r\n* Type hints for all functions\r\n\r\n* Fix type of `*shares`\r\n\r\n* Update setup.cfg\r\n\r\n* Update tests/sympc/approximations/sigmoid_test.py\r\n\r\n* Update tanh.py\r\n\r\n* Update static.py\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-06-04T18:30:04+01:00",
          "tree_id": "d0d0efaaa1a1fa74e7587a031241844e4812c359",
          "url": "https://github.com/OpenMined/SyMPC/commit/6b17043088aac9939957ce038eb49d40b6fd302f"
        },
        "date": 1622828002504,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.10770120059303334,
            "unit": "iter/sec",
            "range": "stddev: 0.052045552826054",
            "extra": "mean: 9.284947563200006 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "3aa4703451665f63fca888fe9e53d106828c624c",
          "message": "added tests and fixed GradPow, GradMatMul (#211)\n\n* added tests and fixed GradPow, GradMatMul\r\n\r\n* added exception case for pow\r\n\r\n* added seprate tests\r\n\r\n* foxed tests",
          "timestamp": "2021-06-05T08:53:00+01:00",
          "tree_id": "944968ac1ad7a58929bf7dec0372dd9a93c54860",
          "url": "https://github.com/OpenMined/SyMPC/commit/3aa4703451665f63fca888fe9e53d106828c624c"
        },
        "date": 1622880113736,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.11814659910165741,
            "unit": "iter/sec",
            "range": "stddev: 0.0466382874547533",
            "extra": "mean: 8.464060816000005 sec\nrounds: 5"
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
          "id": "70e6eaf1eb6a7d58bb0cd580a5211b063e7b18e5",
          "message": "Test for RSTensor send and get. Also added equal operation for RSTensor (#207)\n\n* Test for RSTensor send and get. Also added equal operation for RSTensor\r\n\r\n* Precommit hook\r\n\r\n* Modified RSTensor Test and equal operator - Session UUID\r\n\r\n* minor comment  fix\r\n\r\n* Modified Tests to incorporate uuid\r\n\r\n* Do not skip send and get tests:\r\n\r\nCo-authored-by: rasswanth-s <43314053+rasswanth-s@users.noreply.github.com>",
          "timestamp": "2021-06-05T14:20:43+05:30",
          "tree_id": "1006c16fce71841375ac3afcdf4514fe6ccfc536",
          "url": "https://github.com/OpenMined/SyMPC/commit/70e6eaf1eb6a7d58bb0cd580a5211b063e7b18e5"
        },
        "date": 1622883227562,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1189469298880625,
            "unit": "iter/sec",
            "range": "stddev: 0.06880293646855977",
            "extra": "mean: 8.407110641199996 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}