#ifndef HST_GENERATOR_HPP
#define HST_GENERATOR_HPP

#include <random>

namespace hst
{
    class Generator
    {
    public:
        Generator() = default;
        ~Generator() = default;

        void sample(uint32_t sampleNb, uint32_t distributionSize, int *data)
        {
            std::uniform_int_distribution<std::mt19937::result_type> distribution{0, distributionSize - 1};
            for (std::size_t i = 0; i < sampleNb; i++)
                data[i] = distribution(m_generator);
        }

    private:
        std::random_device m_device;
        std::mt19937 m_generator;
    };
}

#endif // HST_GENERATOR_HPP