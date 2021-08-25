window.BENCHMARK_DATA = {
  "lastUpdate": 1629901476811,
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
          "id": "9ae54136f42b7fcfeafccd0aa0162539b832a49b",
          "message": "Modified allowlist to use absolute paths. (#252)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting",
          "timestamp": "2021-06-25T21:01:29+05:30",
          "tree_id": "eda59d5be5cc6af636de1776ced123c94d7b0bc4",
          "url": "https://github.com/OpenMined/SyMPC/commit/9ae54136f42b7fcfeafccd0aa0162539b832a49b"
        },
        "date": 1624635294207,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.11836156107812186,
            "unit": "iter/sec",
            "range": "stddev: 0.13845025626602475",
            "extra": "mean: 8.448688838599997 sec\nrounds: 5"
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
          "id": "e01d4d319140fbf07cfc124d0e21fbe9aa84ee0f",
          "message": "Feat/ci scheduled tests (#253)\n\n* feat: test latest version\r\n\r\n* feat: scheduled tests\r\n\r\n* Update .github/workflows/tests.yml\r\n\r\nCo-authored-by: rasswanth <43314053+rasswanth-s@users.noreply.github.com>\r\n\r\n* feat: schedule 2am UTC\r\n\r\nCo-authored-by: rasswanth <43314053+rasswanth-s@users.noreply.github.com>",
          "timestamp": "2021-06-26T13:52:43+05:30",
          "tree_id": "f2aa3842d0570673bffe6b342e2bf5636b0183a5",
          "url": "https://github.com/OpenMined/SyMPC/commit/e01d4d319140fbf07cfc124d0e21fbe9aa84ee0f"
        },
        "date": 1624695930917,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.1453586415017776,
            "unit": "iter/sec",
            "range": "stddev: 0.04435279548613432",
            "extra": "mean: 6.879535951000003 sec\nrounds: 5"
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
          "id": "a22a4e8bf50e0b256a88c226cfa86edd7ac917a5",
          "message": "Fix gradient not being recorded (#254)\n\n* Fix gradient not being recorded\r\n\r\n* Import order fix\r\n\r\n* Correctly treat return_indices param for maxpool_2d\r\n\r\n* Fix reshape to receive *args",
          "timestamp": "2021-06-27T23:59:07+01:00",
          "tree_id": "824d6c6cb74ea0a5875d4d9e77c37674d8a65d0a",
          "url": "https://github.com/OpenMined/SyMPC/commit/a22a4e8bf50e0b256a88c226cfa86edd7ac917a5"
        },
        "date": 1624834932738,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.12243417221315134,
            "unit": "iter/sec",
            "range": "stddev: 0.1613481335659656",
            "extra": "mean: 8.167654356000003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "theo.leffyr@gmail.com",
            "name": "Théo Ryffel",
            "username": "LaRiffle"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e6f9585fe20e35fb76a172545bf921550fc7d3cb",
          "message": "Fix the crypto provider generation for FSS (#262)",
          "timestamp": "2021-06-28T05:39:24+01:00",
          "tree_id": "82c0f59602458988aeac07154482e09e6ba663fe",
          "url": "https://github.com/OpenMined/SyMPC/commit/e6f9585fe20e35fb76a172545bf921550fc7d3cb"
        },
        "date": 1624855295599,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.5112160468385959,
            "unit": "iter/sec",
            "range": "stddev: 0.016802497062601986",
            "extra": "mean: 1.9561201299999993 sec\nrounds: 5"
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
          "id": "d6411e2dd0871976f97266a335afd2fc9d674dce",
          "message": "ABY3 Truncation Protocol -Semihonest (#257)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* modified names and minor refactor\r\n\r\n* unit test name_changes\r\n\r\n* remove comments aby3",
          "timestamp": "2021-07-01T14:25:54+05:30",
          "tree_id": "ad68c41a7cd33e79dda47593001fb69b7c6d4fc1",
          "url": "https://github.com/OpenMined/SyMPC/commit/d6411e2dd0871976f97266a335afd2fc9d674dce"
        },
        "date": 1625129908583,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.5002104656543103,
            "unit": "iter/sec",
            "range": "stddev: 0.035816102371418225",
            "extra": "mean: 1.999158491599991 sec\nrounds: 5"
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
          "id": "938c1f29a4333374de24aea6344620a53b609b8c",
          "message": "Refactoring RSTensor Share Distribution and Optimised unit tests  (#255)\n\n* Parallelized RSTensor Share Distribution\r\n\r\n* Removed 7 & 11 party tests\r\n\r\n* Removed several parties tests to optimize CI compute and time\r\n\r\n* CChanged 3 to 2 parties\r\n\r\n* CChanged 3 to 2 parties\r\n\r\n* Removed parallel execution for share distribution and small changes\r\n\r\n* Modify number of parties\r\n\r\n* Small refractoring\r\n\r\n* Precommit hook\r\n\r\n* Modify test",
          "timestamp": "2021-07-06T17:13:00+01:00",
          "tree_id": "ab070c006ee856bda519437ac0550b399c4dfd6d",
          "url": "https://github.com/OpenMined/SyMPC/commit/938c1f29a4333374de24aea6344620a53b609b8c"
        },
        "date": 1625588138051,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.5006510688788784,
            "unit": "iter/sec",
            "range": "stddev: 0.03321223089184483",
            "extra": "mean: 1.9973991111999965 sec\nrounds: 5"
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
          "id": "ded5b9f1a9692972328259cd7203b7fcca2c263a",
          "message": "Falcon  Multiplication(M) , Matrix Multiplication(S,M) , Beaver Triple Generation.(S-Semi-honest,M-Malicious) (#264)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Modified type annotations\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* modified triple reconstruction",
          "timestamp": "2021-07-10T05:26:15+05:30",
          "tree_id": "1a311a3d859fd4aa491a1b274f0478e2f7c4d0ee",
          "url": "https://github.com/OpenMined/SyMPC/commit/ded5b9f1a9692972328259cd7203b7fcca2c263a"
        },
        "date": 1625875141025,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.48632322507219056,
            "unit": "iter/sec",
            "range": "stddev: 0.02281229003677964",
            "extra": "mean: 2.0562456170000076 sec\nrounds: 5"
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
          "id": "090c11dbefbebaab160c6480ade2d3df120715a1",
          "message": "Add sigmoid to torch.nn.functional (#277)\n\n* Add sigmoid to torch.nn.functional\r\n\r\n* precommit hook",
          "timestamp": "2021-07-15T21:39:02+01:00",
          "tree_id": "3df5a728785174cdeecf77557abd06332945f56b",
          "url": "https://github.com/OpenMined/SyMPC/commit/090c11dbefbebaab160c6480ade2d3df120715a1"
        },
        "date": 1626381680657,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.5382827492790234,
            "unit": "iter/sec",
            "range": "stddev: 0.028566118301189596",
            "extra": "mean: 1.8577597022000076 sec\nrounds: 5"
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
          "id": "5596b791ffd81d0d48abdb02033bcbfb23e5b35a",
          "message": "Extension of ReplicatedSharedTensor to Binary,Prime order rings. (#269)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Modified type annotations\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* update change from malicious_mult\r\n\r\n* extended rst to binary,prime\r\n\r\n* modified modulus session\r\n\r\n* modified session przs and rst distribution\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* Added tests and mul for prime,binary\r\n\r\n* fix ring_issue\r\n\r\n* made PRIME_NUMBER global and added tests for ring_size in session\r\n\r\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\r\n\r\n* modified aby3 tests\r\n\r\n* modified type annotations\r\n\r\n* added tests for add,sub,mul for prime rings\r\n\r\n* modified triple reconstruction\r\n\r\n* linting\r\n\r\n* modified mul to take session spcific config\r\n\r\n* modified type annotations and space\r\n\r\n* modified random generation in session and trunc algo randomness",
          "timestamp": "2021-07-20T06:29:27+05:30",
          "tree_id": "06fabf9485a73b34200b18a714a6f4d1b512a909",
          "url": "https://github.com/OpenMined/SyMPC/commit/5596b791ffd81d0d48abdb02033bcbfb23e5b35a"
        },
        "date": 1626742935356,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.45985884769891766,
            "unit": "iter/sec",
            "range": "stddev: 0.037288274905730674",
            "extra": "mean: 2.1745803195999995 sec\nrounds: 5"
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
          "id": "5a5f44223196cb2a4fecf199eb87aa10c95a43a6",
          "message": "Allow additional attributes from torch modules into SyMPC modules (#280)\n\n* init conv2d\r\n\r\n* Some chamges\r\n\r\n* moved additional attributes to module level\r\n\r\n* Added docs and precommit\r\n\r\n* Precommit hook\r\n\r\n* Added a test to check for attribute",
          "timestamp": "2021-07-24T16:56:07+05:30",
          "tree_id": "fda258ecb463d9885cf463b44a51f774fa47d488",
          "url": "https://github.com/OpenMined/SyMPC/commit/5a5f44223196cb2a4fecf199eb87aa10c95a43a6"
        },
        "date": 1627126127836,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.48284135349101326,
            "unit": "iter/sec",
            "range": "stddev: 0.05101482509644849",
            "extra": "mean: 2.071073640999998 sec\nrounds: 5"
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
          "id": "af178dbc99cd1077871346113a4a05abbb510ffc",
          "message": "Falcon: Added XOR (#275)\n\n* Changed relative paths to absolute paths\n\n* Linting\n\n* Added truncation\n\n* Changed parties\n\n* Linting\n\n* Added ABY3 Protocol folder\n\n* modified to trunc1 algorithm\n\n* Added more tests\n\n* Modified Falcon Tests\n\n* Added malicious mult with truncation\n\n* Refactored and modified tests\n\n* added tests and made random_gen global\n\n* Minor refactoring\n\n* modfied redistribution and added tests\n\n* revert distribution and modify type annotations\n\n* modified resharing\n\n* Added support for tensor pointer\n\n* Increased test coverage\n\n* Modified truncation algorithm name\n\n* Added triple verfication and mask\n\n* ABY3 refactoring changes\n\n* Malicious mult refactored-green\n\n* modified to aby3 name_changes\n\n* Falcon malicious mult -check-modification\n\n* modify spdz to session attribute\n\n* Modify spdz to use session and linting\n\n* modified crypto primitive provider tests\n\n* modify prrs tests\n\n* remove malicious not implemented\n\n* added tests\n\n* Added beaver test and reduced round complexity\n\n* modified przs shape and ops\n\n* Added matmul\n\n* Modified type annotations\n\n* modified return type name\n\n* changes kwargs format\n\n* changed kwargs type for beaver\n\n* update change from malicious_mult\n\n* extended rst to binary,prime\n\n* modified modulus session\n\n* modified session przs and rst distribution\n\n* minor refactor\n\n* minor refactor\n\n* revert prrs encoding\n\n* Modified session tests and added test for malicious behavious in mul\n\n* Added tests and mul for prime,binary\n\n* fix ring_issue\n\n* made PRIME_NUMBER global and added tests for ring_size in session\n\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\n\n* modified aby3 tests\n\n* modified type annotations\n\n* added tests for add,sub,mul for prime rings\n\n* modified triple reconstruction\n\n* linting\n\n* modified mul to take session spcific config\n\n* modified type annotations and space\n\n* modified random generation in session and trunc algo randomness\n\n* added xor\n\n* added docs\nstrings\n\n* typo\n\n* moves xor to apply_ops\n\n* fixes\n\n* add type bool to _get_shape\n\n* added ex for xor\n\n* add support for xor at MPC level\n\n* pre commit\n\n* fixed issues\n\n* added new test\n\nCo-authored-by: rasswanth-s <43314053+rasswanth-s@users.noreply.github.com>",
          "timestamp": "2021-07-26T18:59:31+01:00",
          "tree_id": "900a35511215756a21cc31b77ed2a0d4d4d0ad2b",
          "url": "https://github.com/OpenMined/SyMPC/commit/af178dbc99cd1077871346113a4a05abbb510ffc"
        },
        "date": 1627322540199,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.46760048909627094,
            "unit": "iter/sec",
            "range": "stddev: 0.034734481864908524",
            "extra": "mean: 2.1385777459999984 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hershdhillon23@gmail.com",
            "name": "Hersh Dhillon",
            "username": "hershd23"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "38b81177093942e20f56f91a0a669725dfdda8ba",
          "message": "Reordered flaky tests and ordered all the tests sequentially (#289)",
          "timestamp": "2021-08-01T15:18:05+01:00",
          "tree_id": "dc418a47dd748f818937826b2c9d3c47b852f1f6",
          "url": "https://github.com/OpenMined/SyMPC/commit/38b81177093942e20f56f91a0a669725dfdda8ba"
        },
        "date": 1627827648493,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.47554586950709193,
            "unit": "iter/sec",
            "range": "stddev: 0.032289577146766335",
            "extra": "mean: 2.102846568799998 sec\nrounds: 5"
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
          "id": "fe863dbaa42d31b5a8e1477527c1fc8cec290fc4",
          "message": "ABY3 : Bit Injection and local decomposition. (#266)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Added bit injection skel\r\n\r\n* Modified type annotations\r\n\r\n* added ring_size_from_type\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* update change from malicious_mult\r\n\r\n* extended rst to binary,prime\r\n\r\n* modified modulus session\r\n\r\n* modified session przs and rst distribution\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* Added tests and mul for prime,binary\r\n\r\n* fix ring_issue\r\n\r\n* made PRIME_NUMBER global and added tests for ring_size in session\r\n\r\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\r\n\r\n* modified aby3 tests\r\n\r\n* Revert \"added ring_size_from_type\"\r\n\r\nThis reverts commit 988403f087d041f3b86deb59cdcc711d86302037.\r\n\r\n* Revert \"Added bit injection skel\"\r\n\r\nThis reverts commit 9dffeac23e298f570d61328538ad7b49051e45bd.\r\n\r\n* modified type annotations\r\n\r\n* added tests for add,sub,mul for prime rings\r\n\r\n* modified triple reconstruction\r\n\r\n* Added bit_injection and local_decomposition\r\n\r\n* linting\r\n\r\n* modified mul to take session spcific config\r\n\r\n* modified bit_injection to changes from modulus PR\r\n\r\n* added bit injection tests\r\n\r\n* modified aby3 bit injection test\r\n\r\n* deep copy share elements\r\n\r\n* modified to use list comprehension\r\n\r\n* modified type annotations and space\r\n\r\n* modified random generation in session and trunc algo randomness\r\n\r\n* linting\r\n\r\n* modified sanity checks and added tests for local decomposition\r\n\r\n* modified to use xor\r\n\r\n* modified tests\r\n\r\n* increased coverage\r\n\r\n* comment and name changes\r\n\r\n* removed usage of zip\r\n\r\n* decompose bug fix\r\n\r\n* hardcoded test, and modified spdz eps,delta, global variable for nr_parites\r\n\r\n* modified spdz bug\r\n\r\n* minor change",
          "timestamp": "2021-08-02T22:08:34+01:00",
          "tree_id": "090ae3cc72e9214146a3f3cea9422d32d74a0e44",
          "url": "https://github.com/OpenMined/SyMPC/commit/fe863dbaa42d31b5a8e1477527c1fc8cec290fc4"
        },
        "date": 1627938689285,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.8869153705317961,
            "unit": "iter/sec",
            "range": "stddev: 0.04133295798418396",
            "extra": "mean: 1.127503292 sec\nrounds: 5"
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
          "id": "92fdc17c060b0c78d8788addc76d7b2c219848e1",
          "message": "Falcon : Select Shares (#273)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Added bit injection skel\r\n\r\n* Modified type annotations\r\n\r\n* added ring_size_from_type\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* update change from malicious_mult\r\n\r\n* extended rst to binary,prime\r\n\r\n* modified modulus session\r\n\r\n* modified session przs and rst distribution\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* Added tests and mul for prime,binary\r\n\r\n* fix ring_issue\r\n\r\n* made PRIME_NUMBER global and added tests for ring_size in session\r\n\r\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\r\n\r\n* modified aby3 tests\r\n\r\n* Revert \"added ring_size_from_type\"\r\n\r\nThis reverts commit 988403f087d041f3b86deb59cdcc711d86302037.\r\n\r\n* Revert \"Added bit injection skel\"\r\n\r\nThis reverts commit 9dffeac23e298f570d61328538ad7b49051e45bd.\r\n\r\n* modified type annotations\r\n\r\n* added tests for add,sub,mul for prime rings\r\n\r\n* modified triple reconstruction\r\n\r\n* Added bit_injection and local_decomposition\r\n\r\n* linting\r\n\r\n* modified mul to take session spcific config\r\n\r\n* modified bit_injection to changes from modulus PR\r\n\r\n* added bit injection tests\r\n\r\n* modified aby3 bit injection test\r\n\r\n* added select shares and tests.\r\n\r\n* deep copy share elements\r\n\r\n* modified to use list comprehension\r\n\r\n* modified type annotations and space\r\n\r\n* modified type annotations\r\n\r\n* modified to work on tensor inputs\r\n\r\n* added todo for crypto provider\r\n\r\n* modified random generation in session and trunc algo randomness\r\n\r\n* linting\r\n\r\n* modified sanity checks and added tests for local decomposition\r\n\r\n* modified comments\r\n\r\n* modified to use xor\r\n\r\n* modified tests\r\n\r\n* increased coverage\r\n\r\n* increased coverage\r\n\r\n* comment and name changes\r\n\r\n* removed usage of zip\r\n\r\n* decompose bug fix\r\n\r\n* hardcoded test, and modified spdz eps,delta, global variable for nr_parites\r\n\r\n* modified spdz bug\r\n\r\n* minor change\r\n\r\n* minor changes\r\n\r\n* hardcoded test of select shares\r\n\r\n* Fix Syft commit\r\n\r\n* reverted back to dev\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-08-07T06:58:35+05:30",
          "tree_id": "333384ca1f145d285d9c1fdadec6a40a3245ca5f",
          "url": "https://github.com/OpenMined/SyMPC/commit/92fdc17c060b0c78d8788addc76d7b2c219848e1"
        },
        "date": 1628299846473,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.0979013629500425,
            "unit": "iter/sec",
            "range": "stddev: 0.06761375670068664",
            "extra": "mean: 910.8286351999936 msec\nrounds: 5"
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
          "id": "ec281fa644a39e9d039d2bf0e5e2cd92471b217d",
          "message": "Add GradDiv to Autograd  (#284)\n\n* Random changes\r\n\r\n* Add DivGrad\r\n\r\n* Add GradDiv\r\n\r\n* Remove accidental commit\r\n\r\n* Precommit\r\n\r\n* Complete GradDiv\r\n\r\n* Remoe blank line\r\n\r\n* Add GradDiv test\r\n\r\n* Remove pytest skip\r\n\r\n* Precoomit hook\r\n\r\n* Reduced atol, introduced new test and removed reshape\r\n\r\n* precommit\r\n\r\n* Added caching and 1/y\r\n\r\n* Fixed reciprocal function\r\n\r\n* precommit hook\r\n\r\n* Removed multiple ops test\r\n\r\n* Little refractor\r\n\r\n* Precommit\r\n\r\n* Remove accidental test skip\r\n\r\n* Add parameters to slot and remove dtype conversion in gradadd\r\n\r\n* Casted to torch tensor\r\n\r\n* changed conv slots to tuple\r\n\r\n* Some changes\r\n\r\n* Cast inputs to torch tensor in grad functions: sum, add,sub and div\r\n\r\n* Removed unecessary torch casting\r\n\r\n* Small changes",
          "timestamp": "2021-08-07T20:10:34+01:00",
          "tree_id": "60f68af2346ce85075ac0f1e0335d7d89b826fd0",
          "url": "https://github.com/OpenMined/SyMPC/commit/ec281fa644a39e9d039d2bf0e5e2cd92471b217d"
        },
        "date": 1628363593380,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.1290432703155118,
            "unit": "iter/sec",
            "range": "stddev: 0.06562529152576789",
            "extra": "mean: 885.705646800011 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "saxena.shubhank.19@gmail.com",
            "name": "Shubhank Saxena",
            "username": "shubhank-saxena"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "093b3ae444cb2d715c388f6fd4f6c24cec13323f",
          "message": "Add codecov.yml file (#296)\n\n* Add codecov.yml file\r\n\r\n* Add codecov comments on PR\r\n\r\n* Increase range of codecov\r\n\r\n* minor change\r\n\r\nCo-authored-by: rasswanth-s <43314053+rasswanth-s@users.noreply.github.com>",
          "timestamp": "2021-08-08T12:40:40+05:30",
          "tree_id": "425f2d98a44ab3db7a7cbf78d1aa8fbb6b411e0b",
          "url": "https://github.com/OpenMined/SyMPC/commit/093b3ae444cb2d715c388f6fd4f6c24cec13323f"
        },
        "date": 1628406803851,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 0.8723507469595189,
            "unit": "iter/sec",
            "range": "stddev: 0.04210888603019284",
            "extra": "mean: 1.1463279001999922 sec\nrounds: 5"
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
          "id": "a5d197d7f19477e5652c049764659c32f38a34f1",
          "message": "Use get directly (#298)\n\n* Use get directly\n\n* Add session check back",
          "timestamp": "2021-08-08T13:24:49+01:00",
          "tree_id": "638375d7f162e80b0000d3c3c22f5752d8797c84",
          "url": "https://github.com/OpenMined/SyMPC/commit/a5d197d7f19477e5652c049764659c32f38a34f1"
        },
        "date": 1628425621721,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.4649537311152456,
            "unit": "iter/sec",
            "range": "stddev: 0.0073386954642775575",
            "extra": "mean: 682.6154155999973 msec\nrounds: 5"
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
          "id": "1e45f57d458b2036cfeaf401bb2da913453c6733",
          "message": "Separate Github runners for compute intensive tests. (#299)\n\n* created separate runner for compute intensive tests\r\n\r\n* name_change\r\n\r\n* modified order\r\n\r\n* minor name change\r\n\r\n* split as four runners\r\n\r\n* minor change",
          "timestamp": "2021-08-08T21:14:26+01:00",
          "tree_id": "0252755eb198bbfeb5a24e8ca770baee095e5733",
          "url": "https://github.com/OpenMined/SyMPC/commit/1e45f57d458b2036cfeaf401bb2da913453c6733"
        },
        "date": 1628453798774,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.456366179181686,
            "unit": "iter/sec",
            "range": "stddev: 0.010419075573161131",
            "extra": "mean: 686.6404989999751 msec\nrounds: 5"
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
          "id": "7330b547fdcb0738675bc08cf48aa5d5512772f0",
          "message": "Separate Runner for Linting. (#304)\n\n* added separate workflow for linting\r\n\r\n* removed pre-commit from test runners\r\n\r\n* bump pre-commit version",
          "timestamp": "2021-08-12T06:26:10+01:00",
          "tree_id": "e00639b7419cac711a0a3b5c2827d835c368e06c",
          "url": "https://github.com/OpenMined/SyMPC/commit/7330b547fdcb0738675bc08cf48aa5d5512772f0"
        },
        "date": 1628746112757,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.4431657943873086,
            "unit": "iter/sec",
            "range": "stddev: 0.008159859122556422",
            "extra": "mean: 692.9210793999914 msec\nrounds: 5"
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
          "id": "92f7f96193a2c1f491fe959caa2e0f797168c5e7",
          "message": "ABY3: Bit Decomposition Orchestrator Implementation (#285)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Added bit injection skel\r\n\r\n* Modified type annotations\r\n\r\n* added ring_size_from_type\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* update change from malicious_mult\r\n\r\n* extended rst to binary,prime\r\n\r\n* modified modulus session\r\n\r\n* modified session przs and rst distribution\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* Added tests and mul for prime,binary\r\n\r\n* fix ring_issue\r\n\r\n* made PRIME_NUMBER global and added tests for ring_size in session\r\n\r\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\r\n\r\n* modified aby3 tests\r\n\r\n* Revert \"added ring_size_from_type\"\r\n\r\nThis reverts commit 988403f087d041f3b86deb59cdcc711d86302037.\r\n\r\n* Revert \"Added bit injection skel\"\r\n\r\nThis reverts commit 9dffeac23e298f570d61328538ad7b49051e45bd.\r\n\r\n* modified type annotations\r\n\r\n* added tests for add,sub,mul for prime rings\r\n\r\n* modified triple reconstruction\r\n\r\n* Added bit_injection and local_decomposition\r\n\r\n* linting\r\n\r\n* modified mul to take session spcific config\r\n\r\n* modified bit_injection to changes from modulus PR\r\n\r\n* added bit injection tests\r\n\r\n* modified aby3 bit injection test\r\n\r\n* added select shares and tests.\r\n\r\n* deep copy share elements\r\n\r\n* modified to use list comprehension\r\n\r\n* modified type annotations and space\r\n\r\n* modified type annotations\r\n\r\n* modified to work on tensor inputs\r\n\r\n* added todo for crypto provider\r\n\r\n* modified random generation in session and trunc algo randomness\r\n\r\n* added bit decomposition\r\n\r\n* modified typo\r\n\r\n* modified parallel execution\r\n\r\n* linting\r\n\r\n* orchestrator ttp\r\n\r\n* modified sanity checks and added tests for local decomposition\r\n\r\n* modified comments\r\n\r\n* modified to use xor\r\n\r\n* modified tests\r\n\r\n* increased coverage\r\n\r\n* increased coverage\r\n\r\n* modified local decomposition\r\n\r\n* comment and name changes\r\n\r\n* minor format\r\n\r\n* added TODO comments\r\n\r\n* fixed typo\r\n\r\n* removed usage of zip\r\n\r\n* decompose bug fix\r\n\r\n* modified for bitwise bit injection\r\n\r\n* hardcoded test, and modified spdz eps,delta, global variable for nr_parites\r\n\r\n* modified spdz bug\r\n\r\n* minor change\r\n\r\n* linting\r\n\r\n* minor changes\r\n\r\n* hardcoded test of select shares\r\n\r\n* Fix Syft commit\r\n\r\n* hardcoded tests\r\n\r\n* added comment for unused code\r\n\r\n* added sanity checks in bit extraction and right shift\r\n\r\n* added separate workflow for linting\r\n\r\n* removed pre-commit from test runners\r\n\r\n* bump pre-commit version\r\n\r\n* modified variable names\r\n\r\n* modified bit extraction to list comphrehension\r\n\r\n* minor change -review\r\n\r\n* modified bit decomposition to use starmap\r\n\r\n* minor changes\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-08-14T13:05:34+01:00",
          "tree_id": "eccccb323e5a2979443de6193e66ab81b47d6bcb",
          "url": "https://github.com/OpenMined/SyMPC/commit/92f7f96193a2c1f491fe959caa2e0f797168c5e7"
        },
        "date": 1628942893740,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.1033049583437127,
            "unit": "iter/sec",
            "range": "stddev: 0.011573442061553972",
            "extra": "mean: 906.367720399993 msec\nrounds: 5"
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
          "id": "250397273272a727621b1699856966dee0a98f62",
          "message": "Falcon : Private compare (#274)\n\n* Changed relative paths to absolute paths\r\n\r\n* Linting\r\n\r\n* Added truncation\r\n\r\n* Changed parties\r\n\r\n* Linting\r\n\r\n* Added ABY3 Protocol folder\r\n\r\n* modified to trunc1 algorithm\r\n\r\n* Added more tests\r\n\r\n* Modified Falcon Tests\r\n\r\n* Added malicious mult with truncation\r\n\r\n* Refactored and modified tests\r\n\r\n* added tests and made random_gen global\r\n\r\n* Minor refactoring\r\n\r\n* modfied redistribution and added tests\r\n\r\n* revert distribution and modify type annotations\r\n\r\n* modified resharing\r\n\r\n* Added support for tensor pointer\r\n\r\n* Increased test coverage\r\n\r\n* Modified truncation algorithm name\r\n\r\n* Added triple verfication and mask\r\n\r\n* ABY3 refactoring changes\r\n\r\n* Malicious mult refactored-green\r\n\r\n* modified to aby3 name_changes\r\n\r\n* Falcon malicious mult -check-modification\r\n\r\n* modify spdz to session attribute\r\n\r\n* Modify spdz to use session and linting\r\n\r\n* modified crypto primitive provider tests\r\n\r\n* modify prrs tests\r\n\r\n* remove malicious not implemented\r\n\r\n* added tests\r\n\r\n* Added beaver test and reduced round complexity\r\n\r\n* modified przs shape and ops\r\n\r\n* Added matmul\r\n\r\n* Added bit injection skel\r\n\r\n* Modified type annotations\r\n\r\n* added ring_size_from_type\r\n\r\n* modified return type name\r\n\r\n* changes kwargs format\r\n\r\n* changed kwargs type for beaver\r\n\r\n* update change from malicious_mult\r\n\r\n* extended rst to binary,prime\r\n\r\n* modified modulus session\r\n\r\n* modified session przs and rst distribution\r\n\r\n* minor refactor\r\n\r\n* minor refactor\r\n\r\n* revert prrs encoding\r\n\r\n* Modified session tests and added test for malicious behavious in mul\r\n\r\n* Added tests and mul for prime,binary\r\n\r\n* fix ring_issue\r\n\r\n* made PRIME_NUMBER global and added tests for ring_size in session\r\n\r\n* removed hardcoding of PRIME_NUMBER and moved ring truncation to ABY3\r\n\r\n* modified aby3 tests\r\n\r\n* Revert \"added ring_size_from_type\"\r\n\r\nThis reverts commit 988403f087d041f3b86deb59cdcc711d86302037.\r\n\r\n* Revert \"Added bit injection skel\"\r\n\r\nThis reverts commit 9dffeac23e298f570d61328538ad7b49051e45bd.\r\n\r\n* modified type annotations\r\n\r\n* added tests for add,sub,mul for prime rings\r\n\r\n* modified triple reconstruction\r\n\r\n* Added bit_injection and local_decomposition\r\n\r\n* linting\r\n\r\n* modified mul to take session spcific config\r\n\r\n* modified bit_injection to changes from modulus PR\r\n\r\n* added bit injection tests\r\n\r\n* modified aby3 bit injection test\r\n\r\n* added select shares and tests.\r\n\r\n* added private compare\r\n\r\n* deep copy share elements\r\n\r\n* modified to use list comprehension\r\n\r\n* modified type annotations and space\r\n\r\n* modified type annotations\r\n\r\n* modified to work on tensor inputs\r\n\r\n* modified zp* generation\r\n\r\n* added tests private compare\r\n\r\n* added todo for crypto provider\r\n\r\n* modified private compare algo\r\n\r\n* modified random generation in session and trunc algo randomness\r\n\r\n* added bit decomposition\r\n\r\n* modified typo\r\n\r\n* modified parallel execution\r\n\r\n* linting\r\n\r\n* orchestrator ttp\r\n\r\n* modified sanity checks and added tests for local decomposition\r\n\r\n* modified comments\r\n\r\n* modified to use xor\r\n\r\n* modified tests\r\n\r\n* increased coverage\r\n\r\n* increased coverage\r\n\r\n* modified local decomposition\r\n\r\n* comment and name changes\r\n\r\n* minor format\r\n\r\n* added TODO comments\r\n\r\n* fixed typo\r\n\r\n* removed usage of zip\r\n\r\n* decompose bug fix\r\n\r\n* modified for bitwise bit injection\r\n\r\n* hardcoded test, and modified spdz eps,delta, global variable for nr_parites\r\n\r\n* modified spdz bug\r\n\r\n* minor change\r\n\r\n* modified private compare to x>r\r\n\r\n* linting\r\n\r\n* minor changes\r\n\r\n* hardcoded test of select shares\r\n\r\n* Fix Syft commit\r\n\r\n* hardcoded tests\r\n\r\n* added comment for unused code\r\n\r\n* added sanity checks in bit extraction and right shift\r\n\r\n* hardcoded tests in private compare\r\n\r\n* modified tests for float\r\n\r\n* added separate workflow for linting\r\n\r\n* removed pre-commit from test runners\r\n\r\n* bump pre-commit version\r\n\r\n* modified variable names\r\n\r\n* modified bit extraction to list comphrehension\r\n\r\n* typo and list comprehension changes.\r\n\r\n* added exception tests\r\n\r\n* minor style reformats\r\n\r\n* added comments for docstring random generation\r\n\r\n* added comments for exceptions\r\n\r\nCo-authored-by: George Muraru <murarugeorgec@gmail.com>",
          "timestamp": "2021-08-25T19:52:09+05:30",
          "tree_id": "f5b894013e4df5b93cdc0f0845022d736f77c453",
          "url": "https://github.com/OpenMined/SyMPC/commit/250397273272a727621b1699856966dee0a98f62"
        },
        "date": 1629901476170,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/module/module_benchmark_test.py::test_run_inference_conv_model",
            "value": 1.4212017927659133,
            "unit": "iter/sec",
            "range": "stddev: 0.0055956857920891005",
            "extra": "mean: 703.6298469999963 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}