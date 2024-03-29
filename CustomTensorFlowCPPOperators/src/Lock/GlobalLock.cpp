#include "GlobalLock.h"

//==============================================================================================//

std::mutex* GlobalLock::mutex = new std::mutex();

//==============================================================================================//

GlobalLock::GlobalLock()
{

}

//==============================================================================================//

void GlobalLock::getLock()
{
	mutex->lock();
}

//==============================================================================================//

void GlobalLock::releaseLock()
{
	mutex->unlock();
}

//==============================================================================================//
