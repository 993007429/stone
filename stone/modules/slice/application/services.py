from stone.modules.slice.domain.services import SliceDomainService
from stone.seedwork.application.responses import AppResponse


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService):
        self.domain_service = domain_service

    def get_slice_path(self, slice_id: int) -> AppResponse[dict]:
        slice_, message = self.domain_service.get_slice_by_id(slice_id)
        return AppResponse(message=message, data={'slice_path': slice_.slice_path if slice_ else None})

    def upload_slice(self, **kwargs) -> AppResponse[dict]:
        slice_key, slice_filename = self.domain_service.upload_slice(**kwargs)
        return AppResponse(data={'slice_key': slice_key, 'slice_filename': slice_filename})

    def create_slice(self, **kwargs) -> AppResponse[dict]:
        slice_, message = self.domain_service.create_slice(**kwargs)
        if not slice_:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'slice': slice_.dict()})

    def create_label(self, **kwargs) -> AppResponse[dict]:
        label, message = self.domain_service.create_label(**kwargs)
        if not label:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'label': label.dict()})

    def create_dataset(self, **kwargs) -> AppResponse[dict]:
        dataset, message = self.domain_service.create_dataset(**kwargs)
        if not dataset:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'dataset': dataset.dict()})

    def filter_slices(self, **kwargs) -> AppResponse[dict]:
        slices, pagination, message = self.domain_service.filter_slices(**kwargs)
        return AppResponse(message=message, data={'slices': [slice_.dict() for slice_ in slices]}, pagination=pagination)

    def filter_slice_thumbnails(self, **kwargs) -> AppResponse[dict]:
        slices, pagination, message = self.domain_service.filter_slice_thumbnails(**kwargs)
        return AppResponse(message=message, data={'slices': [slice_.dict() for slice_ in slices]}, pagination=pagination)

    def filter_labels(self, **kwargs) -> AppResponse[dict]:
        labels, pagination, message = self.domain_service.filter_labels(**kwargs)
        return AppResponse(message=message, data={'labels': [label.dict() for label in labels]}, pagination=pagination)

    def filter_datasets(self, **kwargs) -> AppResponse[dict]:
        datasets, pagination, message = self.domain_service.filter_datasets(**kwargs)
        return AppResponse(message=message, data={'datasets': [dataset.dict() for dataset in datasets]}, pagination=pagination)

    def get_datasets_with_fuzzy(self, **kwargs) -> AppResponse[dict]:
        datasets, message = self.domain_service.get_datasets_with_fuzzy(**kwargs)
        return AppResponse(message=message, data={'datasets': [dataset.dict() for dataset in datasets]})

    def get_labels_with_fuzzy(self, **kwargs) -> AppResponse[dict]:
        labels, message = self.domain_service.get_labels_with_fuzzy(**kwargs)
        return AppResponse(message=message, data={'labels': [label.dict() for label in labels]})

    def get_slice(self, slice_id: int) -> AppResponse[dict]:
        slice_, message = self.domain_service.get_slice_by_id(slice_id)
        if not slice_:
            return AppResponse(err_code=1, message=message)
        return AppResponse(data={'slice': slice_.dict()})

    def get_label(self, label_id: int) -> AppResponse[dict]:
        label, message = self.domain_service.get_label_by_id(label_id)
        if not label:
            return AppResponse(err_code=1, message=message)
        return AppResponse(data={'label': label.dict()})

    def get_dataset(self, dataset_id: int) -> AppResponse[dict]:
        dataset, message = self.domain_service.get_dataset_by_id(dataset_id)
        if not dataset:
            return AppResponse(err_code=1, message=message)
        return AppResponse(data={'dataset': dataset.dict()})

    def get_dataset_statistics(self, dataset_id: int) -> AppResponse[dict]:
        dataset_statistics, message = self.domain_service.get_dataset_statistics(dataset_id)
        if not dataset_statistics:
            return AppResponse(err_code=1, message=message)
        return AppResponse(data={'dataset_statistics': dataset_statistics.dict()})

    def get_slice_fields(self) -> AppResponse[dict]:
        fields = self.domain_service.get_slice_fields()
        return AppResponse(data={'fields': fields})

    def delete_slices(self, **kwargs) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_slices(**kwargs)
        return AppResponse(message=message, data={'affected_count': deleted_count})

    def update_slices(self, **kwargs) -> AppResponse[dict]:
        updated_count, message = self.domain_service.update_slices(**kwargs)
        return AppResponse(message=message, data={'affected_count': updated_count})

    def add_labels(self, **kwargs) -> AppResponse[dict]:
        affected_count, message = self.domain_service.add_labels(**kwargs)
        return AppResponse(message=message, data={'affected_count': affected_count})

    def add_slices(self, **kwargs) -> AppResponse[dict]:
        affected_count, message = self.domain_service.add_slices(**kwargs)
        return AppResponse(message=message, data={'affected_count': affected_count})

    def remove_slices(self, **kwargs) -> AppResponse[dict]:
        affected_count, message = self.domain_service.remove_slices(**kwargs)
        return AppResponse(message=message, data={'affected_count': affected_count})

    def delete_label(self, label_id: int) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_label(label_id)
        return AppResponse(message=message, data={'affected_count': deleted_count})

    def delete_dataset(self, dataset_id: int) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_dataset(dataset_id)
        return AppResponse(message=message, data={'affected_count': deleted_count})

    def copy_dataset(self, dataset_id: int) -> AppResponse[dict]:
        new_dataset, message = self.domain_service.copy_dataset(dataset_id)
        return AppResponse(message=message, data={'dataset': new_dataset})

    def update_label(self, **kwargs) -> AppResponse[dict]:
        label, message = self.domain_service.update_label(**kwargs)
        if not label:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'label': label.dict()})

    def update_dataset(self, **kwargs) -> AppResponse[dict]:
        dataset, message = self.domain_service.update_dataset(**kwargs)
        if not dataset:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'dataset': dataset.dict()})

    def get_tile(self, **kwargs) -> AppResponse[dict]:
        tile_path, message = self.domain_service.get_tile(**kwargs)
        return AppResponse(message=message, data={'tile_path': tile_path})



