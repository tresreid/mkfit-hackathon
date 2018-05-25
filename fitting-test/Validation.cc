#include "Validation.h"

Validation* Validation::make_validation(const std::string& fileName)
{
  return new Validation();
}

Validation::Validation() {}