// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/utils.h"
#include "utils.h"
#include <sys/time.h>
#include <thread>  
#include <vector>
#define MAX_LOOP 50
#define MT 1

TEST_CASE("Test Brute Force", "[float vector]") {
    using Catch::Approx;

    const int64_t nb = 2000000;
    const int64_t nq = 1;
    const int64_t dim = 512;
    const int64_t k = 1;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::IP );

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
        {knowhere::meta::RADIUS, knowhere::IsMetricType(metric, knowhere::metric::IP) ? 10.0 : 0.99},
    };

    SECTION("Test Search") {
     std::vector<std::thread> threads; 
     queryvar queryvar1(train_ds,query_ds,conf);
     faiss::BaseData::getState().store(faiss::BASE_DATA_STATE::MODIFIED);       
     struct timeval t1,t2;
     double timeuse;
     gettimeofday(&t1,NULL);
#ifdef MT
     for (int i = 0; i < MAX_LOOP; i++)
     {
	    threads.emplace_back(WrapSearch, queryvar1);
	//auto res = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
     }
     for (auto& thread : threads) {
        thread.join();
    }

#else
    auto res = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
#endif
     gettimeofday(&t2,NULL);
     timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

     std::cout << "所有线程已完成执行。" << std::endl;

    }
#if 0
    SECTION("Test Search With Buf") {
        auto ids = new int64_t[nq * k];
        auto dist = new float[nq * k];

        auto start = std::chrono::high_resolution_clock::now();

        auto res = knowhere::BruteForce::SearchWithBuf<float>(train_ds, query_ds, ids, dist, conf, nullptr);

       auto end = std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
       std::cout << "Section duration: " << duration.count() << " microseconds" << std::endl;

 
	REQUIRE(res == knowhere::Status::success);
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * k] == i);
            if (metric == knowhere::metric::IP) {
                REQUIRE(dist[i * k] == 0);
            } else {
                REQUIRE(std::abs(dist[i * k] - 1.0) < 0.00001);
            }
        }
        delete[] ids;
        delete[] dist;
    }

    SECTION("Test Range Search") {

        auto start = std::chrono::high_resolution_clock::now();
    
	auto res = knowhere::BruteForce::RangeSearch<float>(train_ds, query_ds, conf, nullptr);

	auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Section duration: " << duration.count() << " microseconds" << std::endl;


	REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        auto lims = res.value()->GetLims();
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(lims[i] == (size_t)i);
            REQUIRE(ids[i] == i);
            if (metric == knowhere::metric::IP) {
                REQUIRE(dist[i] == 0);
            } else {
                REQUIRE(std::abs(dist[i] - 1.0) < 0.00001);
            }
        }
    }
#endif
}

