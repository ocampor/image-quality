# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. 

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Local Development

The project has a `Makefile` to easily configure the environment and run the tests. To build the
development image just run in your console:
```.env
make build-images
```

and to execute the tests
```.env
make run-test
```

## Pull Request Process

1. Branch from the develop branch and, if needed, rebase to the current develop branch before submitting
   your pull request. If it doesn't merge cleanly with develop you may be asked to rebase your changes.
1. Commits should be as small as possible, while ensuring that each commit is correct independently
   (i.e., each commit should compile and pass tests).
1. Update the README.md with details of changes to the interface, this includes new environment 
   variables, exposed ports, useful file locations and container parameters.
1. Increase the version numbers in any examples files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
1. You may merge the Pull Request in once you have the sign-off of one project maintainer.
