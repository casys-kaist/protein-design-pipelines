#!/usr/bin/env bash
docker pull quay.io/nf-core/proteinfold_esmfold:1.1.1
docker tag quay.io/nf-core/proteinfold_esmfold:1.1.1 esmfold:latest