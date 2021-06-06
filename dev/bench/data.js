window.BENCHMARK_DATA = {
  "lastUpdate": 1622723394689,
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
      }
    ]
  }
}