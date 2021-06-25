window.BENCHMARK_DATA = {
  "lastUpdate": 1624615197532,
  "repoUrl": "https://github.com/OpenMined/SyMPC",
  "entries": {
    "Pytest-benchmarks": [
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
      },
      {
        "commit": {
          "author": {
            "email": "NiWaRe@users.noreply.github.com",
            "name": "Nicolas Remerscheid",
            "username": "NiWaRe"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3c38813f38d309df8827dd28f2c1354696034956",
          "message": "Add smpc argmax (#173)\n\n* started implementing smpc argmax\r\n\r\n* add lost code\r\n\r\n* Fix remote call\r\n\r\n* Add max argmax\r\n\r\n* Remove Final from typing\r\n\r\n* Remove changes to precommit config\r\n\r\n* Crypten's version of `sigmoid` chebyshev approx\r\n\r\n* WIP Code\r\n\r\n* Fix\r\n\r\n* Docstrings for static\r\n\r\n* cheby->cheby-aliter, cheby-crypten->cheby\r\nSome more docstrings\r\n\r\n* Fix `parallel_execution` args\r\nThis is due to a new PR being merged\r\nFix method name in tanh\r\n\r\n* Move tests to single file\r\n\r\n* Re-add removed APIs, remove redundant changes\r\nTypos\r\n\r\n* Fix cyclic import for typing hinting\r\nhttps://stackoverflow.com/a/39757388/8878627\r\n\r\n* Support equi of torch.Tensor.expand\r\n\r\n* The right way to manage sessions\r\n\r\n* Add expected shape for shares of pairwise\r\n\r\n* Oops, how did you get commited?\r\n\r\n* Fix expected shape\r\n\r\n* Fix tests\r\n\r\n* Change math.prod to np.prod\r\n\r\n* Remove methods that are not called on a remote tensor from api\r\n\r\n* Remove duplicate check\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>\r\nCo-authored-by: Syzygianinfern0 <spsharan2000@gmail.com>",
          "timestamp": "2021-06-05T12:06:04+01:00",
          "tree_id": "2b4c6436239b892ecf1ad9c3cf412d925ebfbdff",
          "url": "https://github.com/OpenMined/SyMPC/commit/3c38813f38d309df8827dd28f2c1354696034956"
        },
        "date": 1622891383250,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.10042138237223654,
            "unit": "iter/sec",
            "range": "stddev: 0.07554276939030835",
            "extra": "mean: 9.958038580799997 sec\nrounds: 5"
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
          "id": "79b51c5f677e4f0537f1cf5c4cfd72c35d0fad51",
          "message": "Add SNYK Token",
          "timestamp": "2021-06-05T13:38:02+01:00",
          "tree_id": "b31f4934442d9b455365b8fd75781de178e474da",
          "url": "https://github.com/OpenMined/SyMPC/commit/79b51c5f677e4f0537f1cf5c4cfd72c35d0fad51"
        },
        "date": 1622896920764,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.10023625473341226,
            "unit": "iter/sec",
            "range": "stddev: 0.07637812269532646",
            "extra": "mean: 9.976430211399997 sec\nrounds: 5"
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
          "id": "5144a24779a72778c22a2a539b2b5e1aca4f2991",
          "message": "Add Softmax (#215)\n\n* Add softmax boilerplate\r\nTODO: Implement torch.sum\r\n\r\n* Increase robustness of tests\r\nSquash a few bugs\r\n\r\n* Type hints in docstring\r\n\r\n* Remove unnecessary import\r\nI think\r\n\r\n* 0*tensor -> przs",
          "timestamp": "2021-06-07T08:40:10+01:00",
          "tree_id": "c00f86d4cc4671b9ac8015966b34ac4fdffa20c7",
          "url": "https://github.com/OpenMined/SyMPC/commit/5144a24779a72778c22a2a539b2b5e1aca4f2991"
        },
        "date": 1623051786246,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1251229192825807,
            "unit": "iter/sec",
            "range": "stddev: 0.042441768227393456",
            "extra": "mean: 7.992140894200008 sec\nrounds: 5"
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
          "id": "a96797d62ecfd4f7e70a476a37dd45113f710ee4",
          "message": "Added security_type attribute to protocol (#222)\n\n* Added security_type attribute to protocol\r\n\r\n* fix typos and rename fss tests",
          "timestamp": "2021-06-07T20:27:24+05:30",
          "tree_id": "39e8e62b85af7fff791755fe5a725bfffcab8ff5",
          "url": "https://github.com/OpenMined/SyMPC/commit/a96797d62ecfd4f7e70a476a37dd45113f710ee4"
        },
        "date": 1623080974692,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1056705220910013,
            "unit": "iter/sec",
            "range": "stddev: 0.12177699204897714",
            "extra": "mean: 9.463377110400007 sec\nrounds: 5"
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
          "id": "a4c83dd9ce936788cf53156503f17021ca83edee",
          "message": "Add len method (#224)",
          "timestamp": "2021-06-07T21:56:13+05:30",
          "tree_id": "2855bc5fc7ffdb9e7216ea703bcab9f7ef0871cf",
          "url": "https://github.com/OpenMined/SyMPC/commit/a4c83dd9ce936788cf53156503f17021ca83edee"
        },
        "date": 1623083432978,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.10868810178160283,
            "unit": "iter/sec",
            "range": "stddev: 0.06556179828684972",
            "extra": "mean: 9.200639109599996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "me@madhavajay.com",
            "name": "Madhava Jay",
            "username": "madhavajay"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "16fa17ab61fc37f0aba2f565bda812ae55eb1cdc",
          "message": "torch 1.8.1 (#225)\n\n* Bumped up to allow torch==1.8.1 torchcsprng==0.2.1\r\n\r\n* Fixing issue with torchcsprng==0.2.1 on Windows\r\n\r\n* Install Torch CPU builds for windows first\r\n\r\n* Fixed didnt move shell in workflow\r\n\r\n* Install CPU builds on linux and windows for all torch\r\n\r\n* Renaming step in CI",
          "timestamp": "2021-06-08T18:05:58+05:30",
          "tree_id": "22d9407c43f578db10480c332f9c3c1e60324621",
          "url": "https://github.com/OpenMined/SyMPC/commit/16fa17ab61fc37f0aba2f565bda812ae55eb1cdc"
        },
        "date": 1623155976200,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12200083151038006,
            "unit": "iter/sec",
            "range": "stddev: 0.1025780001968753",
            "extra": "mean: 8.196665445799999 sec\nrounds: 5"
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
          "id": "f2c2804d77c8579e1cfd0b33877e660749d4b2cf",
          "message": "Distribute shares and reconstruct functionality for RSTensor  (#220)\n\n* Initialize distribute and reconstruct for RSTensor\r\n\r\n* Better docstrings\r\n\r\n* Update check for FPEncoder\r\n\r\n* Precommit hook\r\n\r\n* Moved test to RSTensor\r\n\r\n* Allow base=1 and precision=0\r\n\r\n* Added number of shares test and small changes\r\n\r\n* Fix test and add session uiud\r\n\r\n* Add session uiud and fix tests\r\n\r\n* fix get_shares parameter\r\n\r\n* FIx tests\r\n\r\n* Precommit hook\r\n\r\n* Modify protocol definition in tests\r\n\r\n* Refractored distribute shares code\r\n\r\n* Precommit hook\r\n\r\n* Malicious mode of reconstruction\r\n\r\n* Make tests pass\r\n\r\n* Added docstrings and type hints\r\n\r\n* Added tests for malicious security exception, getitem and setitem for RSTensor and some refractoring\r\n\r\n* Fix reconstruct function\r\n\r\n* Fixed test\r\n\r\n* Basic refractoring\r\n\r\n* Removed some blank lines\r\n\r\n* Readd security level check\r\n\r\n* Small changes for cleaner code\r\n\r\n* Correct setitem in RSTensor\r\n\r\n* Update .gitignore\r\n\r\n* Update .gitignore\r\n\r\n* Update docstrings\r\n\r\n* Small variable name change\r\n\r\n* Rename less parties falcon test",
          "timestamp": "2021-06-11T08:44:10+05:30",
          "tree_id": "27b55f078d981ae7d55545d516803110090ea4f6",
          "url": "https://github.com/OpenMined/SyMPC/commit/f2c2804d77c8579e1cfd0b33877e660749d4b2cf"
        },
        "date": 1623381434452,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.13372902120998356,
            "unit": "iter/sec",
            "range": "stddev: 0.3498397806044645",
            "extra": "mean: 7.477808414000004 sec\nrounds: 5"
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
          "id": "d84302dd1f70db9048edb5b0f4d8779f959f2d04",
          "message": "Custom value for pytest workers (#226)\n\n* Added custom value\r\n\r\n* Incorporate Shares VM's for tests\r\n\r\n* Deleted node store clear\r\n\r\n* Revert Shared VM's\r\n\r\n* Included pytest-order for ordering of flaky tests\r\n\r\n* Adding -x flag ,to stop tests after first failure\r\n\r\n* Removed Windows CI",
          "timestamp": "2021-06-11T19:49:55+05:30",
          "tree_id": "ae03e03f7922e7d833da68d3b6cea30f562672d8",
          "url": "https://github.com/OpenMined/SyMPC/commit/d84302dd1f70db9048edb5b0f4d8779f959f2d04"
        },
        "date": 1623421411006,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1106714551111216,
            "unit": "iter/sec",
            "range": "stddev: 0.06454463253883358",
            "extra": "mean: 9.035753609599988 sec\nrounds: 5"
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
          "id": "19b86b8d47f2d69a9c9c5f431bf1ca89951346f0",
          "message": "Add debug info and reduce tolerance (#234)",
          "timestamp": "2021-06-11T20:43:12+05:30",
          "tree_id": "37683d80e7c469c11963fd27ba3e6842cb2354bc",
          "url": "https://github.com/OpenMined/SyMPC/commit/19b86b8d47f2d69a9c9c5f431bf1ca89951346f0"
        },
        "date": 1623424568262,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.13684603825229286,
            "unit": "iter/sec",
            "range": "stddev: 0.24496456575509384",
            "extra": "mean: 7.307482282799993 sec\nrounds: 5"
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
          "id": "4cb3359f4f8199bafcdaf10ea5aa766744288a5e",
          "message": "added GradReLU function (#238)",
          "timestamp": "2021-06-12T12:39:49+01:00",
          "tree_id": "c03b6f465b4e0c0415db8bcb35f55a91cf9c7dec",
          "url": "https://github.com/OpenMined/SyMPC/commit/4cb3359f4f8199bafcdaf10ea5aa766744288a5e"
        },
        "date": 1623498159505,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1415980907042587,
            "unit": "iter/sec",
            "range": "stddev: 0.021641820246167544",
            "extra": "mean: 7.062242118 sec\nrounds: 5"
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
          "id": "309ab87edf6f2251839ca909d503069d477566f7",
          "message": "added forward test (#232)\n\n* added forward tests\r\n\r\n* assured not flakiness",
          "timestamp": "2021-06-12T18:27:50+05:30",
          "tree_id": "baff7a3279d735d31a2c0eb5ed4f514c97a7b900",
          "url": "https://github.com/OpenMined/SyMPC/commit/309ab87edf6f2251839ca909d503069d477566f7"
        },
        "date": 1623502851214,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1356863133000414,
            "unit": "iter/sec",
            "range": "stddev: 0.03952077046795081",
            "extra": "mean: 7.369940089599993 sec\nrounds: 5"
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
          "id": "09a62f5f096855e569fbe776061fad71c84b23ec",
          "message": "Implementation of PRRS(Pseudo-Random Random Share) (#228)\n\n* Added PRRS to Session\r\n\r\n* Added tests for PRRS and PRZS\r\n\r\n* Fix typos\r\n\r\n* Modified Docstrings -CTR mode",
          "timestamp": "2021-06-12T20:43:34+05:30",
          "tree_id": "3ebaf689beef573bb74a9c3836ae0c6d1a6b4c1e",
          "url": "https://github.com/OpenMined/SyMPC/commit/09a62f5f096855e569fbe776061fad71c84b23ec"
        },
        "date": 1623511010594,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12395660984491694,
            "unit": "iter/sec",
            "range": "stddev: 0.13660206987240467",
            "extra": "mean: 8.067339057199996 sec\nrounds: 5"
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
          "id": "70987f139932054da53fec8e0bf146d4ecfad4be",
          "message": "Fixed reconstruction when torch matrices are a secret and added tests to verify same  (#241)\n\n* Fixed reconstruction when torch matrices are a secret and added tests to verify same\r\n\r\n* Fixed bug in malicious reconstruction\r\n\r\n* Improved tests with mixed matrices\r\n\r\n* Remove print statements in test\r\n\r\n* Add deterministic secret and seperate float and matrix tests",
          "timestamp": "2021-06-13T22:30:32+05:30",
          "tree_id": "4d128e6f231695cc98d7c6254a8915e748b434a4",
          "url": "https://github.com/OpenMined/SyMPC/commit/70987f139932054da53fec8e0bf146d4ecfad4be"
        },
        "date": 1623603809259,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.13296588090879294,
            "unit": "iter/sec",
            "range": "stddev: 0.07195388971228597",
            "extra": "mean: 7.520726318400006 sec\nrounds: 5"
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
          "id": "2005f3393c03296bbdf18cecafc9c0582783e84f",
          "message": "Implementation of add/sub operations -RSTensor (#242)\n\n* Added add/sub operations -RSTensor\r\n\r\n* fix typo and modify docstring\r\n\r\n* Added tests for two MPCTensor\r\n\r\n* Added local RSTensor tests\r\n\r\n* changes test names\r\n\r\n* change rank format\r\n\r\n* Refactored and added tests for different share class",
          "timestamp": "2021-06-13T22:36:00+05:30",
          "tree_id": "3c45a7e86d83ebf978b8524e6fd0d860ba1ae9d6",
          "url": "https://github.com/OpenMined/SyMPC/commit/2005f3393c03296bbdf18cecafc9c0582783e84f"
        },
        "date": 1623604145257,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.13277628262666558,
            "unit": "iter/sec",
            "range": "stddev: 0.09339408177527164",
            "extra": "mean: 7.531465561599998 sec\nrounds: 5"
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
          "id": "4531945ff37016157cb90676f6fc370923fd0f7b",
          "message": "added backward function test for MPC Tensor (#229)\n\n* added backward test\r\n\r\n* added test without grad\r\n\r\n* added new tests\r\n\r\n* imporoved tests\r\n\r\n* resolve conflicts",
          "timestamp": "2021-06-16T20:01:05+05:30",
          "tree_id": "b476fa8e55771f9f10bf77e4d41de44e1200f630",
          "url": "https://github.com/OpenMined/SyMPC/commit/4531945ff37016157cb90676f6fc370923fd0f7b"
        },
        "date": 1623854059539,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12490856169292656,
            "unit": "iter/sec",
            "range": "stddev: 0.06715659992415246",
            "extra": "mean: 8.005856335600004 sec\nrounds: 5"
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
          "id": "ff18bf4ebc41214aa573b730c1b8ad9568795407",
          "message": "Max pooling 2D (#221)\n\n* Add max_pool2d forward and backwards\r\n\r\n* Remove arg that would be the result a tuple\r\n\r\n* Update src/sympc/module/nn/functional.py\r\n\r\nCo-authored-by: Anubhav Raj Singh <anubhavraj.08@gmail.com>\r\n\r\n* Update src/sympc/tensor/grads/grad_functions.py\r\n\r\nCo-authored-by: Anubhav Raj Singh <anubhavraj.08@gmail.com>\r\n\r\n* Separate function for sanity check\r\n\r\n* Actually return value from sanity check\r\n\r\n* Decrease the minimum value\r\n\r\n* Remove test for input padding\r\n\r\n* Add order\r\n\r\n* Add order\r\n\r\n* Use reconstruct not get\r\n\r\n* Fix typo in comment\r\n\r\n* Reduce maxpool more\r\n\r\n* Fix comments\r\n\r\nCo-authored-by: Anubhav Raj Singh <anubhavraj.08@gmail.com>",
          "timestamp": "2021-06-17T18:01:34+01:00",
          "tree_id": "b2dea89f08e68d439fb5f50d4d4c559c1de06df9",
          "url": "https://github.com/OpenMined/SyMPC/commit/ff18bf4ebc41214aa573b730c1b8ad9568795407"
        },
        "date": 1623949483185,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.13491216444818369,
            "unit": "iter/sec",
            "range": "stddev: 0.11788824970307439",
            "extra": "mean: 7.4122300542 sec\nrounds: 5"
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
          "id": "ee3827cbdff3baac5bbfe005536fe435633014ce",
          "message": "Remove deprecated syft.load (#246)\n\n* Remove deprecated syft.load\r\n\r\n* trigger GitHub actions",
          "timestamp": "2021-06-22T14:06:07+05:30",
          "tree_id": "fa77f7685eb3530e1a5b29190002f8f2cf730240",
          "url": "https://github.com/OpenMined/SyMPC/commit/ee3827cbdff3baac5bbfe005536fe435633014ce"
        },
        "date": 1624351175678,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1176311551190353,
            "unit": "iter/sec",
            "range": "stddev: 0.04768485549038625",
            "extra": "mean: 8.501149197999997 sec\nrounds: 5"
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
          "id": "e790dd651beca63f2594368e60c85388e85868ba",
          "message": "Modified RSTensor to use parallel execution (#247)\n\n* Modified RSTensor to use parallel execution\r\n\r\n* minor refactor\r\n\r\n* Added sanity check for share_ptrs and tests\r\n\r\n* Fix typo",
          "timestamp": "2021-06-24T16:36:50+05:30",
          "tree_id": "8c886ff6d0922bba9158b7b913dc7be334b33ad5",
          "url": "https://github.com/OpenMined/SyMPC/commit/e790dd651beca63f2594368e60c85388e85868ba"
        },
        "date": 1624533001051,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.127349807465465,
            "unit": "iter/sec",
            "range": "stddev: 0.1175754624578919",
            "extra": "mean: 7.852387215200008 sec\nrounds: 5"
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
          "id": "153238fd7b3b8ad3f30e988bd899f418dfe606db",
          "message": "Falcon Semi-honest Integer multiplication operation  (#243)\n\n* Initialized Falcon multiplication operation, refractored RSTensor and MPCTensor\r\n\r\n* Added malicious security test and small refractoring in test\r\n\r\n* Added integer matrix and value test. Also removed unecessary op_str argument in RSTensor sanity check\r\n\r\n* Update src/sympc/protocol/falcon/falcon.py\r\n\r\nCo-authored-by: Anubhav Raj Singh <anubhavraj.08@gmail.com>\r\n\r\n* Minor refractoring\r\n\r\n* Run precommit\r\n\r\n* Update src/sympc/tensor/replicatedshare_tensor.py\r\n\r\nCo-authored-by: rasswanth <43314053+rasswanth-s@users.noreply.github.com>\r\n\r\n* Precommit hook\r\n\r\n* Used PRZS for random value generation and changed shape value\r\n\r\n* Small refractoring\r\n\r\n* Little refractoring\r\n\r\n* Small changes for cleaner code\r\n\r\n* Some changes\r\n\r\n* Some refractoring\r\n\r\n* Fixed bug\r\n\r\n* Added parallel execution of resharing and small refractoring\r\n\r\n* Added chceck for number of parties\r\n\r\n* added check for 3 parties\r\n\r\n* Parallelized share multiplication and small refractoring\r\n\r\n* Combined parallelized execution into one\r\n\r\n* Modified parallel execution wih debug statements\r\n\r\n* Modified parallel execution wih debug statements\r\n\r\n* Parallelized Falcon mul sucessfully\r\n\r\n* Added type annotation and initiated alt session in sanity check\r\n\r\n* Added type annotation and initiated alt session in sanity check\r\n\r\n* Made PRZS masking inside parallel function\r\n\r\n* Made PRZS masking inside parallel function\r\n\r\n* Remove blank line\r\n\r\n* Remove blank lines\r\n\r\n* PRecommit hook\r\n\r\n* Add comment\r\n\r\n* Remove blank lines\r\n\r\n* Refractored sanity check\r\n\r\n* Small refractoring\r\n\r\n* Small changes\r\n\r\n* Remove shape from allowlist\r\n\r\nCo-authored-by: Anubhav Raj Singh <anubhavraj.08@gmail.com>\r\nCo-authored-by: rasswanth <43314053+rasswanth-s@users.noreply.github.com>",
          "timestamp": "2021-06-25T15:26:41+05:30",
          "tree_id": "ad3393a91e569e4e3377434bfb32313db8c1820a",
          "url": "https://github.com/OpenMined/SyMPC/commit/153238fd7b3b8ad3f30e988bd899f418dfe606db"
        },
        "date": 1624615196939,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12767175081850718,
            "unit": "iter/sec",
            "range": "stddev: 0.168212343506652",
            "extra": "mean: 7.8325862502 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}